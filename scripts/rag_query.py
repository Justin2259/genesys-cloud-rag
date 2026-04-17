#!/usr/bin/env python3
"""
Query the Genesys Cloud RAG pipeline.

Usage:
    python scripts/rag_query.py --query "what happens when caller presses 2"
    python scripts/rag_query.py --query "voicemail routing" --chunk-type dynamic_group_routing
    python scripts/rag_query.py --query "main menu options" --flow "My IVR" --top-k 3
    python scripts/rag_query.py --query "data tables used" --verbose
    python scripts/rag_query.py --query "what changed recently" --chunk-type flow_change
    python scripts/rag_query.py --query "recording policy" --chunk-type recording_policy
"""
import sys
import json
import time
import argparse
import subprocess
import urllib.request
import urllib.error

VPS_HOST = "root@your.vps.host.ip"
SSH_KEY  = "~/.ssh/your_vps_key"
VPS_PORT = 8765
API_BASE = f"http://localhost:{VPS_PORT}"

CHUNK_TYPES = [
    "flow_overview",
    "menu_option",
    "dynamic_group_routing",
    "task_flow",
    "data_table_reference",
    "queue",
    "group",
    "data_table_schema",
    "prompt",
    "ivr_config",
    "phone_number",
    "schedule_group",
    "schedule",
    "wrap_up_code",
    "flow_metadata",
    "recording_policy",
    "flow_change",
]


def api_call(path: str, method="GET", data=None):
    url = f"{API_BASE}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"API error {e.code}: {body}")


def open_tunnel():
    proc = subprocess.Popen(
        [
            "ssh", "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ExitOnForwardFailure=yes",
            "-N", "-L", f"{VPS_PORT}:localhost:{VPS_PORT}",
            VPS_HOST,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def main():
    parser = argparse.ArgumentParser(description="Query Genesys Cloud RAG pipeline")
    parser.add_argument("--query", required=True, help="Natural language question")
    parser.add_argument("--chunk-type", choices=CHUNK_TYPES, help="Filter by chunk type")
    parser.add_argument("--flow", help="Filter by flow name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Show full chunk text")
    parser.add_argument("--local", action="store_true", help="Skip SSH tunnel (API already reachable)")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Output raw JSON")
    args = parser.parse_args()

    tunnel_proc = None
    if not args.local:
        tunnel_proc = open_tunnel()

    try:
        payload = {
            "query": args.query,
            "top_k": args.top_k,
        }
        if args.chunk_type:
            payload["chunk_type"] = args.chunk_type
        if args.flow:
            payload["flow_name"] = args.flow

        resp = api_call("/query", method="POST", data=payload)

        if args.json_out:
            print(json.dumps(resp, indent=2))
            return

        results = resp.get("results", [])
        print(f"\nQuery: {args.query}")
        if args.chunk_type:
            print(f"Filter chunk_type: {args.chunk_type}")
        if args.flow:
            print(f"Filter flow: {args.flow}")
        print(f"Results: {len(results)}\n")
        print("-" * 60)

        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            score = r.get("score", 0)
            text = r.get("text", "")
            print(f"\n[{i}] score={score:.4f}  type={meta.get('chunk_type')}  flow={meta.get('flow_name')}")
            if args.verbose:
                print()
                for line in text.splitlines():
                    print(f"    {line}")
            else:
                lines = text.splitlines()[:3]
                for line in lines:
                    print(f"    {line}")
                if len(text.splitlines()) > 3:
                    print(f"    ... ({len(text.splitlines()) - 3} more lines, use --verbose)")

        print()

    finally:
        if tunnel_proc:
            tunnel_proc.terminate()


if __name__ == "__main__":
    main()
