# How `build_subgraph` works

`build_subgraph(question)` is the retrieval engine of the whole system. Its job is to take a plain-English question and return a small, relevant slice of the full knowledge graph that the LLM can reason over. The full graph has thousands of nodes — far too large to send to an LLM. This function cuts it down to the ~200 most relevant ones.

---

## Step 0 — How the graph is built before any query runs

`build_subgraph` consumes `graph.json`, but that file has to be created first. It is produced by a two-phase offline build pipeline: `extract.py` then `merge.py`. Understanding this pipeline explains what is actually inside `graph.json` and why.

### Phase 1 — `extract.py`: one Haiku call per email

The raw input is 100,000 Enron emails stored in a CSV inside a zip file. For each email, `extract.py` calls Claude Haiku with a prompt asking it to extract named entities and relationships. Up to 20 emails are processed concurrently (async).

The prompt instructs Haiku to return JSON in this shape:

```json
{
  "nodes": [
    { "id": "Kenneth Lay", "type": "person", "description": "CEO of Enron" },
    { "id": "Enron",       "type": "company", "description": "US energy company" }
  ],
  "edges": [
    { "source": "Kenneth Lay", "target": "Enron", "label": "ceo of" }
  ]
}
```

Haiku sees the email headers (From, To, CC, Subject) plus up to 2000 characters of the body. Every person in the headers is included as a node even if they are not mentioned in the body.

**How many edges can one email produce?**

A single email can produce many edges from two separate sources:

- **Headers** — the From, To, and CC fields. The prompt explicitly mandates that all three become nodes, but it does not explicitly require edges for any of them — edge creation is left to Haiku's judgement about what counts as a "relationship". In practice this breaks down unevenly:

  | Header field | Node guaranteed? | Edge likely? |
  |---|---|---|
  | From | Yes | Yes — Haiku reliably infers sender→recipient "emailed" |
  | To | Yes | Yes — From→To is the most natural relationship in an email, almost always produces an edge |
  | CC | Yes | Inconsistent — Haiku sometimes produces "copied on", sometimes nothing |

  An email with 1 sender, 2 To recipients, and 3 CC'd people could produce anywhere from 2 to 6+ header-derived edges depending on how Haiku interprets the CC field.

- **Body** — Haiku reads the full body text and extracts any relationship it finds. If the body says *"I spoke with Jeff Skilling about the Q3 figures"*, Haiku may produce an edge `Kenneth Lay → Jeff Skilling, "spoke with"` entirely from body content, independent of whether Skilling appeared in any header.

Combined, a single email with a busy CC list and a body mentioning several other people could produce a dozen or more edges in one JSONL record.

This has an important implication for nodes: CC'd people reliably become nodes (the prompt mandates it) but may have no edges at all if neither the headers nor the body gave Haiku enough context to infer a relationship. To recipients are almost always connected via an "emailed" edge; CC recipients may exist in the graph as isolated entities until another email provides a connection.

It also explains why edge **weights** matter after merging. An edge `kenneth lay → enron, "chairman of"` with weight 47 means that specific relationship was independently extracted from 47 different emails. A weight of 2 might mean Haiku made an inference from ambiguous phrasing in just two emails — a much weaker signal.

Each extraction result is immediately written as one line to `build/results/extracted.jsonl`. A JSONL file is just one JSON object per line — it is used here rather than a single large JSON array because it can be written incrementally (if the process crashes at email 60,000, you can resume from there rather than starting over).

Each line in `extracted.jsonl` looks like this:

```json
{
  "index": 4821,
  "file": "maildir/lay-k/inbox/47.",
  "nodes": [
    { "id": "Kenneth Lay",  "type": "person",  "description": "Chairman of Enron" },
    { "id": "Enron",        "type": "company", "description": "US energy company" },
    { "id": "Jeff Skilling","type": "person",  "description": "CEO of Enron" }
  ],
  "edges": [
    { "source": "Kenneth Lay",   "target": "Enron",        "label": "chairman of" },
    { "source": "Jeff Skilling", "target": "Kenneth Lay",  "label": "reported to" }
  ]
}
```

The critical field is `"file"` — the path to the original email within the zip. This is the chunk pointer. Every node extracted from this record is linked back to that source email via this field during the merge phase — though as explained below, most of those links are discarded before `graph.json` is written.

After processing all 100,000 emails, `extracted.jsonl` contains 100,000 lines — one per email.

---

### Phase 2 — `merge.py`: collapsing 100,000 extractions into one graph

The JSONL contains 100,000 independent extractions with overlapping entities. "Kenneth Lay" might appear as a node in 15,000 separate records, sometimes written as "Ken Lay" or "K. Lay". `merge.py` collapses all of this into a single deduplicated graph.

**Normalisation and deduplication**

Every entity name is normalised to lowercase with whitespace collapsed:
```
"Kenneth Lay" → "kenneth lay"
"Ken Lay"     → "ken lay"      ← different normalised key, treated as separate node
```

This normalised string becomes the node's permanent ID. All raw name variants seen across all emails are collected, and the most frequently seen capitalisation wins as the display label.

**Node accumulation**

As each JSONL line is processed, its nodes are merged into running accumulators:

```
node_name_variants["kenneth lay"] → ["Kenneth Lay", "Kenneth Lay", "Ken Lay", ...]
node_type_votes["kenneth lay"]    → {"person": 14823, "role": 3}
node_descriptions["kenneth lay"]  → ["CEO of Enron", "Chairman and CEO", ...]
node_chunks["kenneth lay"]        → ["maildir/lay-k/inbox/47.", "maildir/skilling-j/sent/12.", ...]
```

When the final node is written to the graph, the most-voted type wins, the first description is used, and the chunks list is deduplicated and **capped at 50 entries**:

```python
chunks=list(set(node_chunks[norm_id]))[:50]
```

Not all chunks are stored because `graph.json` is loaded into memory on every server startup and served to the frontend. Storing every chunk path for every node — Kenneth Lay might accumulate 15,000 paths across 100,000 emails — would push the file from a few megabytes to potentially hundreds of megabytes just in chunk arrays. The cap of 50 keeps the file manageable.

The tradeoff is that the majority of source email links are permanently discarded at build time. A node that appeared in 15,000 emails retains pointers to at most 50 of them, and those 50 are in arbitrary order due to the `set()` (covered in the known limitations section of the README). At query time the server collects candidate paths from seed nodes in order, stopping at 100 total (`EMAIL_CANDIDATE_LIMIT`).

**Edge accumulation**

Edges are similarly merged. Every time a `(source, target)` pair is seen, a counter increments and the relationship label is recorded:

```
edge_weights[("kenneth lay", "enron")]  → 47       ← seen in 47 different emails
edge_labels[("kenneth lay", "enron")]   → ["chairman of", "chairman of", "ceo of", ...]
```

The most common label wins. The weight (47) encodes how many emails evidenced this relationship — a higher weight means a stronger, more consistently observed connection.

Crucially, the `file` pointer from each JSONL record is stored on nodes (via `node_chunks`) but **not on edges**. The merge phase discards which specific email each edge instance came from. This means you can trace "Kenneth Lay mentioned in this email" but not "this specific Kenneth Lay → Enron relationship came from this specific email".

**Top-N trimming**

After merging, `merge.py` optionally trims the graph to the top 5000 nodes by degree (number of edges). Nodes that only appeared in one or two emails get dropped, keeping the graph focused on entities with meaningful presence across the corpus.

**Output**

The final `graph.json` is written — a clean, deduplicated graph where every node represents a real-world entity that appeared across many emails, every edge represents an observed relationship with a frequency count, and every node carries pointers back to its source emails.

---

### How JSONL becomes graph.json — the full chain

```
100,000 emails in CSV
        │
        ▼  (extract.py — one Haiku call per email, 20 concurrent)
extracted.jsonl — 100,000 lines, one per email
        │         each line: { index, file, nodes[], edges[] }
        │         "file" is the chunk pointer back to the source email
        ▼  (merge.py — deduplication, frequency counting, trimming)
graph.json — single deduplicated graph
        │    nodes: canonical entities with chunk lists
        │    links: relationships with weights
        ▼
loaded by server.py at startup → used by build_subgraph on every query
```

---

## Step 1 — Load the graph into memory

```python
graph = load_graph()
nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
all_links = graph.get("links", [])
```

### What is `graph.json`?

`graph.json` is the output of `merge.py` — the pre-built knowledge graph for the entire Enron email corpus. It is a single JSON file with two top-level arrays:

```json
{
  "nodes": [
    { "id": "kenneth lay", "label": "Kenneth Lay", "type": "person",
      "description": "CEO and chairman of Enron", "chunks": ["emails/lay/1.txt", ...] },
    { "id": "enron", "label": "Enron", "type": "company",
      "description": "US energy company at centre of accounting scandal", "chunks": [...] }
  ],
  "links": [
    { "source": "kenneth lay", "target": "enron", "label": "chairman of", "weight": 47 },
    { "source": "jeffrey skilling", "target": "enron", "label": "ceo of", "weight": 31 }
  ]
}
```

- **nodes** — named entities extracted from the emails (people, companies, roles, events, locations). Each node has an `id` (the normalised lowercase name used as a key), a human-readable `label`, a short `description`, and `chunks` (a list of email file paths that mentioned this entity).
- **links** — relationships between those entities. Each link has a `source` and `target` node ID, a `label` describing the relationship (e.g. `"reported to"`, `"emailed"`, `"chairman of"`), and a `weight` which is how many emails that relationship was seen in.

This file can be several megabytes and holds thousands of nodes and edges.

### What does `load_graph()` do?

`load_graph()` does three things:

1. **Reads `graph.json` from disk** and parses the JSON into a Python dict. The result is stored in a module-level variable `_graph_cache`. On every subsequent call, the cached copy is returned immediately without re-reading the file — this matters because `build_subgraph` is called on every HTTP request.

2. **Builds the adjacency map** (`_adjacency`) — a dict mapping each node ID to the set of node IDs it is directly connected to:
   ```
   "kenneth lay" → {"enron", "jeffrey skilling", "board of directors", ...}
   "enron"       → {"kenneth lay", "jeffrey skilling", "arthur andersen", ...}
   ```
   This is used in Step 5 to expand from seed nodes to their neighbours without scanning all edges every time.

3. **Builds the keyword index** (`_keyword_index`) — an inverted index from words to node IDs:
   ```
   "chairman" → ["kenneth lay", "board of directors", ...]
   "ceo"      → ["kenneth lay", "jeffrey skilling", ...]
   "enron"    → ["kenneth lay", "enron", "jeffrey skilling", "arthur andersen", ...]
   ```

   **Why it's needed:** the graph has thousands of nodes. Without this index, every query would have to scan every node's text to find matches — slow and wasteful. The index is built once at startup and inverts the relationship: instead of asking "does this node contain the word?", you ask "which nodes contain this word?" and get the answer instantly.

   **What it's built from and how:** the index is constructed in `load_graph()` by looping over every node in `graph.json`:

   ```python
   kidx = defaultdict(list)
   for n in data.get("nodes", []):
       nid = n["id"]
       text = " ".join([n.get("label", ""), n.get("description", "")])
       for word in set(_tokenize(text)):
           kidx[word].append(nid)
   ```

   For a node like:
   ```json
   { "id": "kenneth lay", "label": "Kenneth Lay", "description": "Chairman and former CEO of Enron" }
   ```

   The steps are:
   1. Concatenate label + description → `"Kenneth Lay Chairman and former CEO of Enron"`
   2. Lowercase and strip non-alpha → `["kenneth", "lay", "chairman", "and", "former", "ceo", "of", "enron"]`
   3. Remove stopwords (`"and"`, `"of"`, `"former"`) → `{"kenneth", "lay", "chairman", "ceo", "enron"}`
   4. For each surviving word, append `"kenneth lay"` to that word's entry in `kidx`

   After all nodes are processed:
   ```
   kidx["ceo"]      → ["kenneth lay", "jeffrey skilling", ...]
   kidx["chairman"] → ["kenneth lay", "board of directors", ...]
   kidx["enron"]    → ["kenneth lay", "enron", "jeffrey skilling", ...]
   ```

   The label is the canonical display name chosen by `merge.py` as the most-seen capitalisation across all extractions. The description is the one-line text Haiku wrote for that entity — and crucially, only the **first** description ever seen for that node survives into the graph (the `merge.py` limitation discussed in Step 0).

   This means the keyword index is searching a very thin slice of what the emails actually say — one name and one sentence per entity, regardless of how many emails mention them. If Haiku's first description for Kenneth Lay happened to be `"Enron executive"` rather than `"Chairman and former CEO of Enron"`, the node would never score on queries containing `"ceo"` or `"chairman"` — even though thousands of emails discuss him in those terms.

   **Why "ceo" maps to both Lay and Skilling:** this is not a contradiction. It means the word "ceo" appears in both nodes' description text — Haiku described Skilling as `"CEO of Enron"` and Lay as `"Chairman and former CEO of Enron"` in different emails. The index has no concept of time or role exclusivity, only string matches. Both nodes score equally on the word "ceo", and it is left to the LLM — using edge labels, weights, and email excerpts — to determine who held the role when.

Both `_adjacency` and `_keyword_index` are also cached at module level, so they are only computed once regardless of how many queries come in.

### What is `nodes_by_id`?

`graph.get("nodes", [])` returns the nodes as a **list** — ordered, but with no fast lookup by ID. To find a specific node you would have to scan the entire list from the beginning each time.

The dict comprehension `{n["id"]: n for n in ...}` restructures that list into a dict keyed by `id`:

```
# Before (list — finding "enron" requires scanning every element):
[
  { "id": "kenneth lay", ... },
  { "id": "enron", ... },          # could be at position 4,000
  { "id": "jeffrey skilling", ... }
]

# After (dict — finding "enron" is instant):
{
  "kenneth lay":      { "id": "kenneth lay", ... },
  "enron":            { "id": "enron", ... },
  "jeffrey skilling": { "id": "jeffrey skilling", ... }
}
```

Later in Step 7, every node ID in `node_set` is looked up via `nodes_by_id.get(nid)`. With a list that would mean scanning thousands of nodes for each lookup; with a dict it is a single hash lookup regardless of graph size.

### What is `all_links`?

`graph.get("links", [])` is simply the raw edge list from the JSON — every relationship between every pair of entities across the whole graph. It is held aside unchanged and used in Step 8, where it is filtered down to only the edges where both endpoints landed in the subgraph.

---

## Step 2 — Tokenise the question

```python
question_words = set(_tokenize(question))
```

`_tokenize()` does three things:
1. Lowercases everything
2. Strips out anything that isn't a letter (`re.findall(r"[a-z]+"`)
3. Removes stopwords — common English words like "the", "who", "was", "at", "of" that carry no entity meaning

Example — `"Who was Kenneth Lay's role at Enron?"` becomes:
```
{"kenneth", "lay", "role", "enron"}
```

**Why remove stopwords?** The keyword index maps words to node IDs. Stopwords appear in nearly every sentence. If "the" were kept, it would match every node that has "the" in its description, flooding `scores` with noise and making every node look equally relevant.

**Why use a `set`?** Duplicate words in a question (e.g. "Enron Enron") shouldn't score a node twice for the same question word. The set deduplicates automatically.

---

## Step 3 — Score every node against the question

```python
question_words = set(_tokenize(question))
scores: dict[str, int] = defaultdict(int)
# for each word in the question, find which nodes associated with that word, do for each word 
# the more a certain node appears boost its score
for w in question_words:
    for nid in _keyword_index.get(w, []):
        scores[nid] += 1
```

`_keyword_index` is an inverted index built at startup (in `load_graph()`):
```
"kenneth" → ["kenneth lay", "kenneth rice", ...]
"lay"     → ["kenneth lay", "lay pipes inc", ...]
"enron"   → ["enron", "kenneth lay", "andrew fastow", "enron broadband", ...]
"role"    → ["ceo role", ...]
```

This loop walks every question word, finds all nodes that contain it, and increments those nodes' score by 1.

After the loop for our example:
```
"kenneth lay"    → 2  (matched "kenneth" and "lay")
"enron"          → 1  (matched "enron")
"andrew fastow"  → 1  (matched "enron" in his description)
"kenneth rice"   → 1  (matched "kenneth")
```

**Why this scoring approach?** It is a bag-of-words term frequency count. A node that shares more words with the question is more likely to be what the user is asking about. It is not semantically smart — "CEO" will not match "chief executive" — but it is fast, requires no external model, and works well when questions use the same vocabulary as the entity labels and descriptions.

---

## Step 4 — Pick the top 50 seed nodes

```python
seeds = sorted(scores, key=lambda x: -scores[x])[:SUBGRAPH_SEED_LIMIT]
# SUBGRAPH_SEED_LIMIT = 50
```

Sorts all scored nodes highest-first and takes the top 50. These become the "seeds" — the anchor points from which the subgraph grows.

**Why 50?** It is a balance between recall (enough nodes to cover the question) and noise (too many seeds and you are pulling in unrelated parts of the graph before the 1-hop expansion even starts).

**Why are these called seeds?** Because the next step grows outward from them. They are not the final subgraph — they are starting points.

---

## Step 5 — Expand 1 hop outward from each seed

```python
node_set: set[str] = set(seeds)
for seed in seeds:
    for nb in _adjacency.get(seed, set()):
        node_set.add(nb)
        if len(node_set) >= SUBGRAPH_NEIGHBOUR_LIMIT:
            break
    if len(node_set) >= SUBGRAPH_NEIGHBOUR_LIMIT:
        break
# SUBGRAPH_NEIGHBOUR_LIMIT = 200
```

`_adjacency` is a dict built at startup mapping each node ID to the set of nodes it is directly connected to:
```
"kenneth lay" → {"enron", "jeffrey skilling", "board of directors", ...}
```

For every seed, every node it has a direct edge to is added to `node_set`. The loop stops once `node_set` hits 200 total nodes.

**Why expand to neighbours?** The LLM needs context around the seed nodes to answer questions involving relationships. If the question asks about Kenneth Lay's role, the LLM also needs to see "Enron" and "chairman" nodes and the edges between them — even if those nodes didn't score highly on the keywords. Without expansion, the LLM would see disconnected entities with no relationship structure.

**Why 1 hop only?** 2-hop expansion on a dense graph explodes combinatorially. Starting from 50 seeds, 1-hop already reaches 200 nodes (the cap). 2-hop could reach tens of thousands, which would overflow the context window.

**Why cap at 200?** Every node and edge in `node_set` gets serialised to JSON and pasted into the LLM prompt. 200 nodes produces a prompt of roughly 10,000–30,000 tokens depending on description length. Beyond that, costs rise sharply and the LLM struggles to focus.

---

## Step 6 — Fallback when nothing matches

```python
if not node_set:
    degree = {nid: len(nbrs) for nid, nbrs in _adjacency.items()}
    top = sorted(degree, key=lambda x: -degree[x])[:SUBGRAPH_NEIGHBOUR_LIMIT]
    node_set = set(top)
    seeds = top[:SUBGRAPH_SEED_LIMIT]
```

If no question word matched any node in the index — for example a very abstract question with no named entities — `scores` is empty and `node_set` would be empty too.

The fallback grabs the 200 most-connected nodes in the graph (highest degree). In a knowledge graph built from corporate emails, these tend to be the most prominent entities: the CEO, the company, the key subsidiaries, major events.

**Why degree as a fallback?** Degree centrality correlates with importance in a graph built from entity co-occurrence. Nodes that appear in many edges were mentioned across many emails in many contexts, which makes them a reasonable general-purpose subgraph when there is no query-specific signal to go on.

---

## Step 7 — Strip down node fields

```python
stripped_nodes = [
    {
        "id": n["id"],
        "label": n.get("label", n["id"]),
        "type": n.get("type", ""),
        "description": n.get("description", ""),
    }
    for nid in node_set
    if (n := nodes_by_id.get(nid))
]
```

Each node in `graph.json` also carries `chunk_count` and `chunks` (the email file path pointers). Those are stripped out here because the LLM does not need them — they are only used by `build_email_context()` separately. Sending them to the LLM would waste tokens.

**Why use a walrus operator (`:=`)?** `nodes_by_id.get(nid)` might return `None` if a node ID ended up in `node_set` via adjacency expansion but wasn't in the nodes list (can happen with nodes that appeared only in edges during extraction). The walrus operator assigns and checks in one expression, filtering out those missing nodes cleanly.

---

## Step 8 — Filter edges to only those within the subgraph

```python
stripped_links = [
    {"source": e["source"], "target": e["target"], "label": e.get("label", ""), "weight": e.get("weight", 1)}
    for e in all_links
    if e["source"] in node_set and e["target"] in node_set
]
```

Scans the full edge list and keeps only edges where **both** the source and target node are in `node_set`.

**Why both endpoints?** An edge pointing to a node outside the subgraph would be a dangling reference. The LLM would see a relationship to an entity it has no information about, which is misleading. Requiring both endpoints ensures the subgraph is self-contained.

**Why not filter edges earlier?** You need to know the final `node_set` before you can decide which edges survive. The node set is only fully known after the 1-hop expansion in Step 5.

---

## Step 9 — Return the subgraph and the seed IDs

```python
subgraph = json.dumps({"nodes": stripped_nodes, "links": stripped_links}, ensure_ascii=False)
return subgraph, seeds
```

Returns two things:

- `subgraph` — a JSON string of the filtered nodes and edges, ready to be pasted directly into the LLM prompt as context
- `seeds` — the list of top-scoring node IDs, passed to `build_email_context()` so it knows which nodes to fetch source emails for

**Why return seeds separately rather than all nodes?** Email retrieval is deliberately restricted to seed nodes only. Seeds are the nodes that directly matched the question keywords — they are the most likely to have relevant source emails. Fetching emails for all 200 nodes would pull in a lot of tangentially related content and risk filling the prompt with off-topic evidence.

---

## Step 10 — Fetch the most relevant source emails (`build_email_context`)

`build_subgraph` returns the seed IDs, which are immediately passed to `build_email_context(seed_ids, question)`. This function retrieves the original Enron emails that best support the question and appends them to the LLM prompt as primary source evidence.

### Phase 1 — Collect all candidate chunk paths

```python
candidate_paths: list[str] = []
path_to_label: dict[str, str] = {}
for nid in seed_ids:
    node = nodes_by_id.get(nid)
    for path in node.get("chunks", []):
        if path not in path_to_label:
            candidate_paths.append(path)
            path_to_label[path] = node.get("label", nid)
        if len(candidate_paths) >= EMAIL_CANDIDATE_LIMIT:  # 100
            break
```

Each seed node has a `chunks` list — the email file paths it was extracted from, stored in `graph.json` during the merge phase. This loop collects all chunk paths from all seed nodes (deduplicating across nodes) up to a cap of 100 total candidates. Paths are collected in seed order, so the highest keyword-scoring nodes contribute their chunks first.

**Why 100 candidates and not just take the first few?** The old approach took `[:2]` chunks from each seed node directly — giving at most 10 emails from the first 5 seeds, with no consideration of whether those emails were relevant to the question. 100 candidates gives a wide enough pool to find genuinely relevant emails while staying bounded for the SQLite fetch.

### Phase 2 — Fetch all candidates in one query

```python
placeholders = ",".join("?" * len(candidate_paths))
rows = db.execute(
    f"SELECT file, message FROM emails WHERE file IN ({placeholders})",
    candidate_paths,
).fetchall()
```

All 100 candidate emails are fetched from SQLite in a single `WHERE file IN (...)` query — one round trip regardless of how many candidates there are. This is faster than the previous approach of individual per-file lookups, and we already have everything we need for the scoring step.

### Phase 3 — Score each email by subject line

```python
for file_path, message in rows:
    headers, body = _parse_email_headers(message)
    subject_words = set(_tokenize(headers.get("subject", "")))
    # uses the & operator (a Set Intersection) to find words that appear in both the subject line and the user's question.
    score = len(subject_words & question_words)
    scored.append((score, file_path, headers, body))

scored.sort(key=lambda x: -x[0])
```

For each fetched email, the subject line is tokenised (same stopword removal as the question) and the overlap with the question's words is counted. A subject like `"Re: Enron accounting irregularities"` scores 2 against the question `"how did Enron hide debt?"` (matching `"enron"` and — if present — other terms). A subject like `"Re: office party"` scores 0.

Emails are sorted by score descending and the top `EMAIL_SNIPPET_LIMIT` (10) are taken.

**Why subject lines and not the full body?** The full message is already in memory from Phase 2 — there is no second query. Subject lines are used for scoring because they are a short, dense signal. Scoring the full body would be noisier (bodies are long, repetitive, and full of boilerplate like signatures and forwarding chains) and slower. The subject line captures the email's topic in a few words — exactly what we need to rank relevance quickly.

**The remaining limitation:** subject lines don't always reflect body content. An email with a generic subject like `"FYI"` but a body discussing fraud in detail will score 0 and lose to a less relevant email with a keyword-rich subject. The proper fix is body embeddings, but subject scoring is a significant improvement over the previous arbitrary ordering.

### Phase 4 — Build the snippet strings

```python
for _score, file_path, headers, body in scored[:EMAIL_SNIPPET_LIMIT]:
    label = path_to_label.get(file_path, "")
    snippets.append(
        f"[Entity: {label}] "
        f"From: {headers.get('from','')} To: {headers.get('to','')} "
        f"Subject: {headers.get('subject','')} Date: {headers.get('date','')}\n"
        f"{body[:EMAIL_BODY_LIMIT]}"
    )
```

Each of the top 10 emails is formatted as a snippet — headers (from, to, subject, date) plus the first 400 characters of the body. The `[Entity: label]` tag tells the LLM which graph node this email came from, so it can connect the primary source evidence back to the entity it was retrieved for.

The full snippet block is prepended with `"Source emails (excerpts from the original Enron corpus):"` and inserted into the LLM prompt between the subgraph JSON and the question.

---

## Summary of the full flow

```
question text
     │
     ▼
tokenise → remove stopwords → set of meaningful words
     │
     ▼
for each word → kidx lookup → increment node scores
     │
     ▼
top 50 nodes by score = seeds
     │
     ▼
expand each seed to its 1-hop neighbours → node_set (cap 200)
     │
     ├─► strip node fields → stripped_nodes
     │
     └─► filter all_links to node_set → stripped_links
                    │
                    ▼
          JSON string + seed IDs returned
```

The seeds answer "what is relevant?" — the expansion answers "what is the context around what is relevant?" — the JSON string is what the LLM actually reads.
