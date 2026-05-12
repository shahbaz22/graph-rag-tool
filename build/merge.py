"""
Phase 2: Merge batch results into a NetworkX graph and export graph.json.

Usage:
  python merge.py
  python merge.py --top 5000   # limit to top N nodes by degree for rendering
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import networkx as nx
from tqdm import tqdm

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT = Path(__file__).parent.parent / "graph.json"

VALID_TYPES = {"person", "company", "role", "event", "location"}


def normalize_id(name: str) -> str:
    """Lowercase + strip for deduplication key."""
    return re.sub(r"\s+", " ", name.strip()).lower()


def canonical_name(variants: List[str]) -> str:
    """Pick the most-seen capitalization as the canonical name."""
    counts: Dict[str, int] = defaultdict(int)
    for v in variants:
        counts[v.strip()] += 1
    return max(counts, key=counts.__getitem__)


def load_batch_results() -> List[Dict]:
    results_file = RESULTS_DIR / "extracted.jsonl"
    if not results_file.exists():
        sys.exit(
            f"No results file found at {results_file}. "
            "Run: python extract.py"
        )

    print(f"Loading {results_file} ...")
    all_results = []
    with results_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(all_results):,} records.")
    return all_results


def parse_extraction(result: Dict) -> Tuple[List[Dict], List[Dict], str]:
    """Return (nodes, edges, source_id) from an extraction record."""
    source_id = result.get("file", str(result.get("index", "")))

    nodes = result.get("nodes") or []
    edges = result.get("edges") or []
    for n in nodes:
        if n.get("type") not in VALID_TYPES:
            n["type"] = "person"
    return nodes, edges, source_id


def build_graph(all_results: List[Dict], top_n: Optional[int] = None) -> nx.Graph:
    G = nx.Graph()

    # accumulators for deduplication
    node_name_variants = defaultdict(list)   # norm_id -> raw names
    node_type_votes = defaultdict(lambda: defaultdict(int))
    node_descriptions = defaultdict(list)
    node_chunks = defaultdict(list)
    edge_weights = defaultdict(int)
    edge_labels = defaultdict(list)

    print("Merging extracted entities ...")
    errors = 0
    total_nodes = 0
    total_edges = 0

    for result in tqdm(all_results):
        nodes, edges, custom_id = parse_extraction(result)
        if not nodes and not edges:
            errors += 1
            continue

        total_nodes += len(nodes)
        total_edges += len(edges)

        # index nodes
        local_ids: dict[str, str] = {}  # raw id -> norm_id
        for node in nodes:
            raw_id = node.get("id", "").strip()
            if not raw_id:
                continue
            norm = normalize_id(raw_id)
            local_ids[raw_id] = norm
            node_name_variants[norm].append(raw_id)
            node_type_votes[norm][node.get("type", "person")] += 1
            desc = node.get("description", "")
            if desc:
                node_descriptions[norm].append(desc)
            node_chunks[norm].append(custom_id)

        # index edges
        for edge in edges:
            src_raw = edge.get("source", "").strip()
            tgt_raw = edge.get("target", "").strip()
            label = edge.get("label", "related to").strip()
            if not src_raw or not tgt_raw:
                continue

            # resolve to norm_ids (fallback: normalize on the fly)
            src = local_ids.get(src_raw, normalize_id(src_raw))
            tgt = local_ids.get(tgt_raw, normalize_id(tgt_raw))

            if src == tgt:
                continue

            key = tuple(sorted([src, tgt]))
            edge_weights[key] += 1
            edge_labels[key].append(label)

            # ensure nodes exist even if not in the nodes list
            node_name_variants[src].append(src_raw)
            node_name_variants[tgt].append(tgt_raw)

    print(f"  raw nodes extracted: {total_nodes:,}")
    print(f"  raw edges extracted: {total_edges:,}")
    print(f"  failed/empty results: {errors:,}")

    # build canonical nodes
    print("Building canonical graph ...")
    for norm_id, variants in tqdm(node_name_variants.items()):
        name = canonical_name(variants)
        node_type = max(node_type_votes[norm_id], key=node_type_votes[norm_id].get) if node_type_votes[norm_id] else "person"
        descriptions = node_descriptions[norm_id]
        description = descriptions[0] if descriptions else ""
        G.add_node(
            norm_id,
            label=name,
            type=node_type,
            description=description,
            chunk_count=len(node_chunks[norm_id]),
            chunks=list(set(node_chunks[norm_id]))[:50],  # cap for JSON size
        )

    for (src, tgt), weight in edge_weights.items():
        if src not in G or tgt not in G:
            continue
        # most common label wins
        label_counts = defaultdict(int)
        for lbl in edge_labels[(src, tgt)]:
            label_counts[lbl] += 1
        label = max(label_counts, key=label_counts.__getitem__)
        G.add_edge(src, tgt, label=label, weight=weight)

    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    if top_n and G.number_of_nodes() > top_n:
        print(f"Trimming to top {top_n:,} nodes by degree ...")
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        keep = {node for node, _ in degrees[:top_n]}
        remove = [n for n in G.nodes() if n not in keep]
        G.remove_nodes_from(remove)
        # remove isolated nodes after trim
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
        print(f"After trim: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    return G


def export_graph(G: nx.Graph, output_path: Path):
    data = nx.node_link_data(G)

    # reshape for easier D3 consumption
    nodes = []
    for node in data["nodes"]:
        nodes.append({
            "id": node["id"],
            "label": node.get("label", node["id"]),
            "type": node.get("type", "person"),
            "description": node.get("description", ""),
            "chunk_count": node.get("chunk_count", 0),
            "chunks": node.get("chunks", []),
        })

    links = []
    for link in data["links"]:
        links.append({
            "source": link["source"],
            "target": link["target"],
            "label": link.get("label", ""),
            "weight": link.get("weight", 1),
        })

    output = {
        "metadata": {
            "nodes": len(nodes),
            "edges": len(links),
            "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        },
        "nodes": nodes,
        "links": links,
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False))
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"Exported {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Merge batch results into graph.json")
    parser.add_argument(
        "--top",
        type=int,
        default=5000,
        help="Limit graph to top N nodes by degree (default: 5000)",
    )
    args = parser.parse_args()

    all_results = load_batch_results()
    G = build_graph(all_results, top_n=args.top)
    export_graph(G, OUTPUT)
    print("\nDone. Run:  python ../server.py   to start the demo server.")


if __name__ == "__main__":
    main()
