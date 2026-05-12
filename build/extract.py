"""
Phase 1: Entity extraction via async concurrent Haiku calls.

Note: The LiteLLM proxy does not support the Anthropic Batch API endpoint.
This script uses asyncio + a semaphore to run many concurrent requests instead.

Usage:
  python extract.py                   # Extract all 100k emails
  python extract.py --sample 1000     # Quick test on N emails
  python extract.py --concurrency 30  # Tune concurrency (default: 20)
  python extract.py --resume          # Skip already-processed emails
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import email
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Optional

import anthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "extracted.jsonl"

ZIP_PATH = DATA_DIR / "enron_emails.zip"
SAMPLE_SIZE = 100_000
MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

EXTRACTION_PROMPT = """\
Extract named entities and relationships from the email below. Return ONLY valid JSON — no explanation, no markdown fences.

JSON format:
{{
  "nodes": [{{"id": "Full Formal Name", "type": "person|company|role|event|location", "description": "one-line description"}}],
  "edges": [{{"source": "entity id", "target": "entity id", "label": "relationship verb or phrase"}}]
}}

Rules:
- Normalize names to full formal versions (e.g. "Ken Lay" -> "Kenneth Lay")
- Include every person mentioned in From/To/CC headers as a person node
- Skip trivial filler words; only include meaningful named entities
- Return {{"nodes": [], "edges": []}} if the email has no extractable entities

EMAIL:
{email_text}"""


def parse_email_msg(raw: str) -> Dict:
    msg = email.message_from_string(raw)

    def header(name: str) -> str:
        val = msg.get(name, "") or ""
        return re.sub(r"\s+", " ", val.strip())

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                except Exception:
                    body = str(part.get_payload())
                break
    else:
        try:
            payload = msg.get_payload(decode=True)
            body = payload.decode("utf-8", errors="replace") if payload else str(msg.get_payload())
        except Exception:
            body = str(msg.get_payload())

    return {
        "from": header("From"),
        "to": header("To"),
        "cc": header("CC"),
        "date": header("Date"),
        "subject": header("Subject"),
        "body": body[:2000],
    }


def build_prompt(parsed: Dict) -> str:
    email_text = (
        f"From: {parsed['from']}\n"
        f"To: {parsed['to']}\n"
        f"CC: {parsed['cc']}\n"
        f"Date: {parsed['date']}\n"
        f"Subject: {parsed['subject']}\n\n"
        f"{parsed['body']}"
    )
    return EXTRACTION_PROMPT.format(email_text=email_text.strip())


def load_emails(limit: int) -> List[Dict]:
    csv.field_size_limit(10_000_000)
    print(f"Reading up to {limit:,} emails from {ZIP_PATH} ...")
    emails = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("emails.csv") as f:
            reader = csv.DictReader(
                (line.decode("utf-8", errors="replace") for line in f)
            )
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                parsed = parse_email_msg(row.get("message", ""))
                parsed["file"] = row.get("file", f"email_{i}")
                parsed["index"] = i
                emails.append(parsed)
                if (i + 1) % 10_000 == 0:
                    print(f"  loaded {i + 1:,} ...")
    print(f"Loaded {len(emails):,} emails.")
    return emails


def load_done_indices(resume: bool) -> set:
    if not resume or not RESULTS_FILE.exists():
        return set()
    done = set()
    with RESULTS_FILE.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["index"])
            except Exception:
                pass
    print(f"Resuming: {len(done):,} emails already processed.")
    return done


async def extract_one(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    email_data: Dict,
    out_file,
    lock: asyncio.Lock,
    errors: List,
    retries: int = 3,
):
    async with sem:
        for attempt in range(retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": build_prompt(email_data)}],
                )
                text = response.content[0].text.strip() if response.content else ""
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = {"nodes": [], "edges": []}

                record = {
                    "index": email_data["index"],
                    "file": email_data["file"],
                    "nodes": data.get("nodes", []),
                    "edges": data.get("edges", []),
                }
                async with lock:
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                return

            except anthropic.RateLimitError:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
            except Exception as e:
                if attempt == retries - 1:
                    errors.append((email_data["index"], str(e)))
                else:
                    await asyncio.sleep(1)


async def run_extraction(
    emails: List[Dict],
    concurrency: int,
    resume: bool,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    done = load_done_indices(resume)
    pending = [e for e in emails if e["index"] not in done]

    if not pending:
        print("All emails already processed.")
        return

    print(f"Processing {len(pending):,} emails (concurrency={concurrency}) ...")

    client = anthropic.AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    mode = "a" if resume else "w"
    errors: List = []
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    start = time.time()
    with open(RESULTS_FILE, mode) as out_file:
        tasks = [
            extract_one(client, sem, e, out_file, lock, errors)
            for e in pending
        ]
        await tqdm_asyncio.gather(tasks, desc="Extracting")

    elapsed = time.time() - start
    rate = len(pending) / elapsed if elapsed > 0 else 0
    print(f"\nDone: {len(pending) - len(errors):,} succeeded, {len(errors):,} failed")
    print(f"Time: {elapsed:.0f}s  ({rate:.1f} emails/s)")
    if errors:
        print(f"First 5 errors: {errors[:5]}")


def main():
    parser = argparse.ArgumentParser(description="Enron entity extraction")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE,
                        help=f"Number of emails to process (default: {SAMPLE_SIZE:,})")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests (default: 20)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed emails")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set.")

    emails = load_emails(args.sample)
    asyncio.run(run_extraction(emails, args.concurrency, args.resume))
    print(f"\nResults saved to {RESULTS_FILE}")
    print("Run:  python merge.py   to build graph.json")


if __name__ == "__main__":
    main()
