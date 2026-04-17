#!/usr/bin/env python3
"""
Trigger a Genesys RAG ingest job on the VPS via SSH tunnel and poll until complete.

Usage:
    python scripts/rag_ingest.py --flow "My Main IVR"
    python scripts/rag_ingest.py --flow "My IVR" --reset
    python scripts/rag_ingest.py --org                    # full org ingest
    python scripts/rag_ingest.py --org --reset            # full org ingest, clear first
    python scripts/rag_ingest.py --refresh                # change detection + org entity refresh
    python scripts/rag_ingest.py --flow "My Main IVR" --local  # skip SSH tunnel
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


def ssh_cmd(cmd: str, capture=True) -> str:
    """Run a command on the VPS via SSH."""
    full = ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", VPS_HOST, cmd]
    if capture:
        result = subprocess.run(full, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"SSH error: {result.stderr.strip()}")
        return result.stdout.strip()
    else:
        subprocess.run(full, timeout=30)
        return ""


def api_call(path: str, method="GET", data=None):
    """HTTP call to the RAG API (via SSH tunnel or direct)."""
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
    """Open SSH tunnel to VPS port -> local port."""
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


def poll_job(job_id: str, label: str = "job"):
    """Poll /ingest/{job_id} until complete or failed. Returns summary dict."""
    print(f"\nPolling {label} status...")
    last_msg = ""
    while True:
        status = api_call(f"/ingest/{job_id}")
        current_status = status.get("status")
        msg = status.get("message", "")
        if msg != last_msg:
            print(f"  [{current_status}] {msg}")
            last_msg = msg
        if current_status == "complete":
            return status.get("summary", {})
        elif current_status == "failed":
            print(f"\n[FAIL] {label} failed: {status.get('error')}")
            sys.exit(1)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Trigger Genesys RAG ingest on VPS")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--flow", help="Flow name or UUID to ingest")
    group.add_argument("--org", action="store_true",
                       help="Full org-wide ingest: all flows + all org entities")
    group.add_argument("--refresh", action="store_true",
                       help="Run change detection + org entity refresh (same as weekly job)")
    parser.add_argument("--reset", action="store_true", help="Delete existing chunks before ingest")
    parser.add_argument("--local", action="store_true",
                        help="Assume API already reachable on localhost (no SSH tunnel)")
    args = parser.parse_args()

    tunnel_proc = None
    if not args.local:
        print(f"Opening SSH tunnel to {VPS_HOST}:{VPS_PORT}...")
        tunnel_proc = open_tunnel()

    try:
        try:
            health = api_call("/health")
            print(f"[OK] API reachable. Collection has {health.get('collection_count', '?')} chunks.")
        except Exception as e:
            print(f"[FAIL] API not reachable: {e}")
            sys.exit(1)

        if args.refresh:
            print("\nTriggering change detection + org entity refresh...")
            resp   = api_call("/refresh", method="POST")
            job_id = resp["job_id"]
            print(f"Job started: {job_id}")
            summary = poll_job(job_id, label="refresh")
            print(f"\n[OK] Refresh complete!")
            print(f"  Flows checked:  {summary.get('flows_checked', '?')}")
            print(f"  Flows changed:  {summary.get('flows_changed', '?')}")
            print(f"  Flows skipped:  {summary.get('flows_skipped', '?')}")
            errors = summary.get("errors", [])
            if errors:
                print(f"  Errors ({len(errors)}):")
                for e in errors:
                    print(f"    - {e['flow']}: {e['error']}")

        elif args.org:
            print(f"\nStarting full org ingest...")
            if args.reset:
                print("  (--reset: entire collection will be cleared)")
            resp   = api_call("/ingest", method="POST",
                              data={"flow_name": "__all__", "reset": args.reset})
            job_id = resp["job_id"]
            print(f"Job started: {job_id}")
            summary = poll_job(job_id, label="org ingest")
            print(f"\n[OK] Org ingest complete!")
            print(f"  Total chunks: {summary.get('total_chunks', '?')}")
            print(f"  By type:")
            for chunk_type, count in sorted(summary.get("by_type", {}).items()):
                print(f"    {chunk_type}: {count}")

        else:
            print(f"\nStarting ingest for: {args.flow}")
            if args.reset:
                print("  (--reset: existing chunks will be deleted)")
            resp   = api_call("/ingest", method="POST",
                              data={"flow_name": args.flow, "reset": args.reset})
            job_id = resp["job_id"]
            print(f"Job started: {job_id}")
            summary = poll_job(job_id, label=f"ingest '{args.flow}'")
            print(f"\n[OK] Ingest complete!")
            print(f"  Flow: {summary.get('flow_name')}")
            print(f"  Total chunks: {summary.get('total_chunks')}")
            print(f"  By type:")
            for chunk_type, count in sorted(summary.get("by_type", {}).items()):
                print(f"    {chunk_type}: {count}")

    finally:
        if tunnel_proc:
            tunnel_proc.terminate()


if __name__ == "__main__":
    main()
