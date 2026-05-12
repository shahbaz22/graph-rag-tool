# GraphRAG Knowledge Graph Demo
## Enron Email Dataset — Project Plan
*April 2026*

---

## Overview

An interactive knowledge graph built from the Enron email corpus using Claude Haiku for bulk entity extraction and Claude Sonnet for intelligent querying. The demo renders a D3.js force-directed SVG graph where users can explore the network, ask natural language questions grounded in both graph structure and original email content, and watch the relevant subgraph light up in response.

The graph is the primary interface. The AI is the navigation layer.

**Status: Complete.** All phases implemented and working.

---

## Architecture

### Two-phase design

**Phase 1 — Build (runs once, offline)** ✅
- Downloaded Enron email dataset (517k emails as CSV in a zip)
- Sent each email to Claude Haiku for entity/relationship extraction (async, 20 concurrent)
- Merged and deduplicated into a single NetworkX graph
- Serialized to `graph.json` (~5k nodes, ~50k edges)
- Each node retains `chunks` — file paths pointing back to the source emails

**Phase 2 — Query (runs live in browser)** ✅
- D3.js loads `graph.json` and renders SVG force-directed layout
- User types a question
- Server extracts a relevant subgraph (keyword matching + 1-hop expansion, ~200 nodes)
- Server pulls original email excerpts for the seed nodes from SQLite
- Subgraph + emails sent to Sonnet → structured JSON response streamed back
- D3 highlights the returned nodes/edges, auto-zooms to the area

### Why no embeddings?

The graph structure IS the index. At query time, keyword matching against node labels/descriptions selects seed nodes, then 1-hop expansion captures the relevant neighborhood. This avoids the cost and complexity of a vector store. At ~5k nodes the subgraph extraction runs in milliseconds.

If scaling beyond ~10k nodes, vector search on node descriptions could replace keyword matching for seed selection.

### Email retrieval

Each node in `graph.json` has a `chunks` array of file paths (e.g. `"blair-l/deleted_items/578."`) that map to the `file` column in the original CSV. On first server startup, the CSV is indexed into a SQLite database (`data/emails.db`, 517k rows, 1.6GB). Lookups by file path are O(1).

Emails are used in two ways:
1. **Node panel** — click any node to see its source emails (subject, from, to, date, body)
2. **Query context** — when answering questions, Sonnet receives email excerpts alongside the graph structure, allowing it to cite specific evidence

---

## Tech Stack

| Layer | Tool | Model/Version |
|---|---|---|
| Entity extraction | Claude Haiku | `claude-haiku-4-5-20251001` |
| Graph querying | Claude Sonnet | `claude-sonnet-4-6` (EU endpoint) |
| Graph storage | NetworkX → `graph.json` | — |
| Email storage | SQLite (`data/emails.db`) | Built from CSV on first run |
| Graph rendering | D3.js v7 (SVG) | Force-directed layout |
| Server | Flask (Python) | SSE streaming |
| Frontend | Single HTML file | No build step |

---

## Cost

| Step | Cost |
|---|---|
| Build graph from 100k emails (Haiku, async) | ~$8 |
| Query (Sonnet, ~60k tokens in / ~500 out) | ~$0.02 per query |
| Demo day (100 queries) | ~$2 |
| **Total** | **~$12** |

---

## What was built

### Build pipeline
- `build/extract.py` — async concurrent Haiku calls with `--resume` and `--sample` flags
- `build/merge.py` — entity dedup, name resolution, type voting, edge weight merging, `--top` flag

### Server (`server.py`)
- Flask with 4 endpoints: `/`, `/graph.json`, `/query-stream`, `/emails`
- Keyword index + adjacency map from `graph.json` (in-memory, instant)
- SQLite email index from CSV (1.6GB on disk, built once, O(1) lookups)
- Subgraph extraction: keyword scoring → top 50 seeds → 1-hop expansion (cap 200)
- Email context: up to 10 email excerpts from seed nodes included in Sonnet prompt
- SSE streaming with structured JSON output (answer, details, paths)
- Prompt caching on system prompt

### Frontend (`frontend/index.html`)
- D3.js SVG force-directed graph (replaced canvas-based force-graph library for reliable hover)
- Every node is a `<circle>`, every edge is a `<line>` — native DOM mouse events
- Node sizing by degree, two color modes (connectivity/type), entity type filters
- Hover highlighting with edge labels, zoom controls, node search
- Explore neighborhood (2-hop filter)
- Source email panel with scrollable body and show more/less
- Streaming query UI with progress indicators and structured result rendering
- Slider capped at 800 nodes for SVG performance

---

## Remaining improvements

See `improve.md` for the full prioritized list. Key items not yet implemented:
- Collapsible/resizable sidebar
- Query conversation history
- Fuzzy search
- Edge weight filter slider
- Minimap
- Mobile layout
