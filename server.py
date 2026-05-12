"""
Demo server — serves the frontend and proxies Claude Sonnet queries.

Usage:
  python server.py
  open http://localhost:5000
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

load_dotenv()

app = Flask(__name__, static_folder="frontend")

GRAPH_JSON = Path(__file__).parent / "graph.json"
EMAILS_ZIP = Path(__file__).parent / "data" / "enron_emails.zip"
MODEL = "eu.anthropic.claude-sonnet-4-6"
# Max seed nodes matched by keyword; their 1-hop neighbours are also included
SUBGRAPH_SEED_LIMIT = 50
SUBGRAPH_NEIGHBOUR_LIMIT = 200  # cap total nodes in the subgraph

SYSTEM_PROMPT = """\
You are a knowledge graph analyst specialising in the Enron corporate scandal.
You have access to a subgraph extracted from a knowledge graph built from 100,000 Enron internal emails,
plus excerpts from the original emails that mention the most relevant entities.
Each node is a named entity (person, company, role, event, or location) with an id, label, and type.
Each edge is a directed relationship with a label (e.g. "reported to", "emailed") and weight.

When the user asks a question:
1. Identify the most relevant nodes and edges in the provided subgraph
2. Trace multi-hop paths using ACTUAL node labels and edge labels from the data
3. Call graph_response with your findings — limit nodeIds to the 20 most relevant nodes, edgeIds to the 30 most relevant edges
4. Use **bold** for entity names and → to show relationships in details/paths
5. When email excerpts are provided, cite specific evidence (quotes, dates, senders)"""

_GRAPH_TOOL = {
    "name": "graph_response",
    "description": "Return the graph analysis: highlighted nodes/edges and a plain-English answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "nodeIds": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of the 20 most relevant nodes from the subgraph",
            },
            "edgeIds": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Edge keys in 'source_id|target_id' format, up to 30 most relevant",
            },
            "answer": {
                "type": "string",
                "description": "Concise 1-2 sentence direct answer to the question",
            },
            "details": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-6 bullet findings using **bold** entity names and → for relationships",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key multi-hop paths, e.g. **A** →[rel]→ **B** →[rel]→ **C**",
            },
        },
        "required": ["nodeIds", "edgeIds", "answer", "details", "paths"],
    },
}

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "of","in","on","at","to","for","with","by","from","up","about","into","through",
    "and","or","but","if","as","so","yet","both","either","not","no","nor",
    "who","what","where","when","how","which","that","this","these","those","i",
    "me","my","we","our","you","your","he","his","she","her","it","its","they","their",
}

_client = None        # type: Optional[anthropic.Anthropic]
_graph_cache = None   # type: Optional[dict]
_adjacency = None     # type: Optional[dict]
_keyword_index = None  # type: Optional[dict]
EMAILS_DB = Path(__file__).parent / "data" / "emails.db"
_email_db = None  # type: Optional[sqlite3.Connection]


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY not set. Copy .env.example to .env.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _tokenize(text: str):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


def load_graph() -> dict:
    global _graph_cache, _adjacency, _keyword_index
    if _graph_cache is None:
        if not GRAPH_JSON.exists():
            return {}
        data = json.loads(GRAPH_JSON.read_text())
        _graph_cache = data

        # build adjacency map for 1-hop expansion
        adj = defaultdict(set)
        for e in data.get("links", []):
            s, t = e["source"], e["target"]
            adj[s].add(t)
            adj[t].add(s)
        _adjacency = adj

        # build keyword → [node_id] index
        kidx = defaultdict(list)
        for n in data.get("nodes", []):
            nid = n["id"]
            text = " ".join([n.get("label", ""), n.get("description", "")])
            for word in set(_tokenize(text)):
                kidx[word].append(nid)
        _keyword_index = kidx

    return _graph_cache


def build_subgraph(question: str) -> tuple[str, list[str]]:
    """Return (subgraph_json, seed_node_ids) for the question."""
    graph = load_graph()
    nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    all_links = graph.get("links", [])

    question_words = set(_tokenize(question))
    scores: dict[str, int] = defaultdict(int)
    for w in question_words:
        for nid in _keyword_index.get(w, []):
            scores[nid] += 1

    seeds = sorted(scores, key=lambda x: -scores[x])[:SUBGRAPH_SEED_LIMIT]

    node_set: set[str] = set(seeds)
    for seed in seeds:
        for nb in _adjacency.get(seed, set()):
            node_set.add(nb)
            if len(node_set) >= SUBGRAPH_NEIGHBOUR_LIMIT:
                break
        if len(node_set) >= SUBGRAPH_NEIGHBOUR_LIMIT:
            break

    if not node_set:
        degree = {nid: len(nbrs) for nid, nbrs in _adjacency.items()}
        top = sorted(degree, key=lambda x: -degree[x])[:SUBGRAPH_NEIGHBOUR_LIMIT]
        node_set = set(top)
        seeds = top[:SUBGRAPH_SEED_LIMIT]

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
    stripped_links = [
        {"source": e["source"], "target": e["target"], "label": e.get("label", ""), "weight": e.get("weight", 1)}
        for e in all_links
        if e["source"] in node_set and e["target"] in node_set
    ]
    subgraph = json.dumps({"nodes": stripped_nodes, "links": stripped_links}, ensure_ascii=False)
    return subgraph, seeds


EMAIL_SNIPPET_LIMIT = 10  # max emails to include in prompt
EMAIL_BODY_LIMIT = 400    # chars per email body
EMAIL_CANDIDATE_LIMIT = 100  # max chunk paths to fetch before subject scoring


def _parse_email_headers(msg: str) -> tuple[dict, str]:
    header_end = msg.find("\n\n")
    if header_end == -1:
        header_end = len(msg)
    headers = {}
    for line in msg[:header_end].split("\n"):
        if ": " in line:
            k, v = line.split(": ", 1)
            kl = k.lower().strip()
            if kl in ("from", "to", "subject", "date"):
                headers[kl] = v.strip()
    body = msg[header_end:].strip()
    return headers, body


def build_email_context(seed_ids: list[str], question: str) -> str:
    """Pull the most question-relevant email snippets from seed node chunks.

    Collects all chunk paths across seed nodes (up to EMAIL_CANDIDATE_LIMIT),
    fetches them in one query, scores each by subject-line keyword overlap with
    the question, and returns the top EMAIL_SNIPPET_LIMIT emails.
    """
    db = get_email_db()
    if not db:
        return ""
    graph = load_graph()
    nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    question_words = set(_tokenize(question))

    # Collect all chunk paths from seed nodes up to candidate limit
    candidate_paths: list[str] = []
    path_to_label: dict[str, str] = {}
    for nid in seed_ids:
        node = nodes_by_id.get(nid)
        if not node:
            continue
        label = node.get("label", nid)
        for path in node.get("chunks", []):
            if path not in path_to_label:
                candidate_paths.append(path)
                path_to_label[path] = label
            if len(candidate_paths) >= EMAIL_CANDIDATE_LIMIT:
                break
        if len(candidate_paths) >= EMAIL_CANDIDATE_LIMIT:
            break

    if not candidate_paths:
        return ""

    # Fetch all candidates in one query
    placeholders = ",".join("?" * len(candidate_paths))
    rows = db.execute(
        f"SELECT file, message FROM emails WHERE file IN ({placeholders})",
        candidate_paths,
    ).fetchall()

    if not rows:
        return ""

    # Score each email by subject-line keyword overlap with the question
    scored = []
    for file_path, message in rows:
        headers, body = _parse_email_headers(message)
        subject_words = set(_tokenize(headers.get("subject", "")))
        score = len(subject_words & question_words)
        scored.append((score, file_path, headers, body))

    scored.sort(key=lambda x: -x[0])

    snippets = []
    for _score, file_path, headers, body in scored[:EMAIL_SNIPPET_LIMIT]:
        label = path_to_label.get(file_path, "")
        snippets.append(
            f"[Entity: {label}] "
            f"From: {headers.get('from','')} To: {headers.get('to','')} "
            f"Subject: {headers.get('subject','')} Date: {headers.get('date','')}\n"
            f"{body[:EMAIL_BODY_LIMIT]}"
        )

    if not snippets:
        return ""
    return "Source emails (excerpts from the original Enron corpus):\n\n" + "\n---\n".join(snippets)


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/graph.json")
def graph_json():
    if not GRAPH_JSON.exists():
        return jsonify({"error": "graph.json not found. Run build/merge.py first."}), 404
    return send_file(str(GRAPH_JSON), mimetype="application/json")


@app.route("/query", methods=["POST"])
def query():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    graph = load_graph()
    if not graph:
        return jsonify({"error": "graph.json not found. Run build/merge.py first."}), 500

    client = get_client()

    subgraph, seeds = build_subgraph(question)
    email_ctx = build_email_context(seeds, question)

    user_content = f"Relevant subgraph (nodes and edges):\n{subgraph}"
    if email_ctx:
        user_content += f"\n\n{email_ctx}"
    user_content += f"\n\nQuestion: {question}"

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            tools=[_GRAPH_TOOL],
            tool_choice={"type": "tool", "name": "graph_response"},
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APIError as e:
        return jsonify({"error": str(e)}), 500

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    result = tool_block.input if tool_block else {"nodeIds": [], "edgeIds": [], "answer": "No result.", "details": [], "paths": []}

    usage = response.usage
    result["usage"] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
    }
    return jsonify(result)


@app.route("/query-stream", methods=["POST"])
def query_stream():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    graph = load_graph()
    if not graph:
        return jsonify({"error": "graph.json not found. Run build/merge.py first."}), 500

    client = get_client()
    subgraph, seeds = build_subgraph(question)
    email_ctx = build_email_context(seeds, question)

    user_content = f"Relevant subgraph (nodes and edges):\n{subgraph}"
    if email_ctx:
        user_content += f"\n\n{email_ctx}"
    user_content += f"\n\nQuestion: {question}"

    def generate():
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                tools=[_GRAPH_TOOL],
                tool_choice={"type": "tool", "name": "graph_response"},
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                response = stream.get_final_message()

            tool_block = next((b for b in response.content if b.type == "tool_use"), None)
            result = tool_block.input if tool_block else {"nodeIds": [], "edgeIds": [], "answer": "No result.", "details": [], "paths": []}

            usage = response.usage
            result["usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            }
            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
        except anthropic.APIError as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


def get_email_db() -> Optional[sqlite3.Connection]:
    global _email_db
    if _email_db is not None:
        return _email_db

    if EMAILS_DB.exists():
        _email_db = sqlite3.connect(str(EMAILS_DB), check_same_thread=False)
        count = _email_db.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
        print(f"Email DB loaded: {count:,} emails.")
        return _email_db

    if not EMAILS_ZIP.exists():
        return None

    # Build SQLite DB from CSV (one-time)
    csv.field_size_limit(10 * 1024 * 1024)
    print(f"Building email DB from CSV → {EMAILS_DB} (one-time)...")
    db = sqlite3.connect(str(EMAILS_DB))
    db.execute("CREATE TABLE emails (file TEXT PRIMARY KEY, message TEXT)")
    batch = []
    with zipfile.ZipFile(EMAILS_ZIP) as z:
        with z.open("emails.csv") as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    batch.append((row[0], row[1]))
                    if len(batch) >= 10000:
                        db.executemany("INSERT OR IGNORE INTO emails VALUES (?,?)", batch)
                        batch.clear()
    if batch:
        db.executemany("INSERT OR IGNORE INTO emails VALUES (?,?)", batch)
    db.commit()
    count = db.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    print(f"Email DB built: {count:,} emails.")
    _email_db = db
    return _email_db


def lookup_emails(file_paths: list[str]) -> list[str]:
    db = get_email_db()
    if not db:
        return []
    results = []
    for fp in file_paths:
        row = db.execute("SELECT message FROM emails WHERE file=?", (fp,)).fetchone()
        results.append(row[0] if row else "")
    return results


@app.route("/emails", methods=["POST"])
def emails():
    body = request.get_json(force=True)
    chunks = body.get("chunks", [])
    if not chunks:
        return jsonify({"error": "chunks is required"}), 400

    messages = lookup_emails(chunks[:10])
    results = []
    for path, msg in zip(chunks[:10], messages):
        if not msg:
            continue
        headers, body_text = _parse_email_headers(msg)
        results.append({
            "file": path,
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", "(no subject)"),
            "date": headers.get("date", ""),
            "body": body_text[:1000],
        })
    return jsonify({"emails": results})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting GraphRAG demo server on http://localhost:{port}")
    print(f"Graph: {GRAPH_JSON} ({'found' if GRAPH_JSON.exists() else 'NOT FOUND — run build/merge.py'})")
    print(f"Emails: {EMAILS_ZIP} ({'found' if EMAILS_ZIP.exists() else 'NOT FOUND'})")
    get_email_db()
    app.run(host="0.0.0.0", port=port, debug=False)
