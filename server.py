#!/usr/bin/env python3
"""
Genesys Cloud RAG - FastAPI server.
Exposes /health, /query, /ingest, /ingest/{job_id}, /refresh.

Weekly job (every Sunday at midnight UTC):
  - Checks every published flow for version changes
  - For changed flows: diffs old vs new chunks, stores a flow_change chunk in ChromaDB,
    re-ingests new chunks, sends Discord notification
  - Always re-ingests org entities (queues, schedule groups, recording policies, etc.)
"""
import os
import uuid
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

import chromadb
from chromadb.utils import embedding_functions

import requests as _requests

from ingest import (
    _fetch_all, _upsert, _safe_id,
    run_ingest, run_org_ingest,
    run_org_entities_refresh,
    run_flow_change_detection,
    CHROMA_DB_PATH, CHROMA_CACHE_DIR, COLLECTION_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Genesys Cloud RAG", version="2.0.0")

_jobs: dict = {}

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

FULL_CONFIG_TYPES = {
    "inboundcall", "inboundshortmessage", "inboundchat",
    "commonmodule",
    "inqueuecall", "inqueueshortmessage",
    "bot", "digitalbot",
    "outboundcall",
    "voicemail",
    "workflow",
}

# -------------------------------------------------------------------------
# Discord notifications
# -------------------------------------------------------------------------

def _discord(content: str):
    if not DISCORD_WEBHOOK_URL:
        return
    for chunk in [content[i:i+1900] for i in range(0, len(content), 1900)]:
        try:
            _requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=10)
        except Exception as e:
            logger.warning("Discord notify failed: %s", e)


def _discord_change_report(change: dict):
    """Post a Discord message summarising what changed in a flow."""
    flow_name  = change["flow_name"]
    pub_by     = change.get("published_by", "Unknown")
    diff       = change.get("diff", {})

    pub_ts = change.get("published_date", "")
    try:
        dt = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
        pub_ts_fmt = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pub_ts_fmt = pub_ts or "unknown time"

    added_n    = len(diff.get("added", []))
    removed_n  = len(diff.get("removed", []))
    modified_n = len(diff.get("modified", []))

    lines = [
        f"**Flow Changed: {flow_name}**",
        f"Published by: **{pub_by}** | {pub_ts_fmt}",
        f"Version: `{change.get('old_version','?')[:8]}` -> `{change.get('new_version','?')[:8]}`",
        f"Diff: {added_n} added | {modified_n} modified | {removed_n} removed",
    ]

    if diff.get("modified"):
        lines.append("\n**Modified:**")
        for m in diff["modified"][:8]:
            lines.append(f"  - [{m['chunk_type']}] {m['label']}")
        if modified_n > 8:
            lines.append(f"  - ...and {modified_n - 8} more")

    if diff.get("added"):
        lines.append("\n**Added:**")
        for c in diff["added"][:5]:
            label = c.get("task_name", c.get("entity_name", c.get("chunk_id", "?")))
            lines.append(f"  - [{c['chunk_type']}] {label}")
        if added_n > 5:
            lines.append(f"  - ...and {added_n - 5} more")

    if diff.get("removed"):
        lines.append("\n**Removed:**")
        for c in diff["removed"][:5]:
            label = c.get("task_name", c.get("entity_name", c.get("chunk_id", "?")))
            lines.append(f"  - [{c['chunk_type']}] {label}")
        if removed_n > 5:
            lines.append(f"  - ...and {removed_n - 5} more")

    lines.append("\n_Query the RAG for full before/after details._")
    _discord("\n".join(lines))


# -------------------------------------------------------------------------
# flow_change chunk builder
# -------------------------------------------------------------------------

def _build_flow_change_chunk(change: dict) -> dict:
    """
    Build a flow_change chunk from a change detection result.
    Stored in ChromaDB so it is queryable: 'what changed recently?',
    'who published flow X last week?', etc.
    """
    flow_name   = change["flow_name"]
    flow_id     = change["flow_id"]
    pub_by      = change.get("published_by", "Unknown")
    pub_date    = change.get("published_date", "")
    old_version = change.get("old_version", "unknown")
    new_version = change.get("new_version", "")
    diff        = change.get("diff", {})

    pub_ts = pub_date
    try:
        dt = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
        pub_ts_fmt = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pub_ts_fmt = pub_ts or "unknown"

    added    = diff.get("added", [])
    removed  = diff.get("removed", [])
    modified = diff.get("modified", [])

    text = (
        f"Flow change record: {flow_name}\n"
        f"Published by: {pub_by}\n"
        f"Published at: {pub_ts_fmt}\n"
        f"Old version: {old_version}\n"
        f"New version: {new_version}\n"
        f"Summary: {len(added)} added, {len(modified)} modified, {len(removed)} removed\n"
    )

    if modified:
        text += "\nModified tasks/items:\n"
        for m in modified:
            text += f"  - [{m['chunk_type']}] {m['label']}\n"
            old_short = m["old_text"][:300].replace("\n", " | ")
            new_short = m["new_text"][:300].replace("\n", " | ")
            text += f"    BEFORE: {old_short}\n"
            text += f"    AFTER:  {new_short}\n"

    if added:
        text += "\nAdded tasks/items:\n"
        for c in added:
            label = c.get("task_name", c.get("entity_name", ""))
            text += f"  - [{c['chunk_type']}] {label}: {c['text'][:200].replace(chr(10), ' | ')}\n"

    if removed:
        text += "\nRemoved tasks/items:\n"
        for c in removed:
            label = c.get("task_name", c.get("entity_name", ""))
            text += f"  - [{c['chunk_type']}] {label}: {c['text'][:200].replace(chr(10), ' | ')}\n"

    chunk_id = _safe_id("flowchange", flow_id, new_version)

    return {
        "chunk_id":       chunk_id,
        "chunk_type":     "flow_change",
        "flow_name":      flow_name,
        "flow_id":        flow_id,
        "published_by":   pub_by,
        "published_date": pub_date,
        "old_version":    old_version,
        "new_version":    new_version,
        "added_count":    len(added),
        "modified_count": len(modified),
        "removed_count":  len(removed),
        "text":           text.strip(),
    }


# -------------------------------------------------------------------------
# Weekly scheduled refresh
# -------------------------------------------------------------------------

def _seconds_until_next_sunday_midnight() -> float:
    now = datetime.now(timezone.utc)
    days_until_sunday = (6 - now.weekday()) % 7
    if days_until_sunday == 0 and (now.hour > 0 or now.minute > 0 or now.second > 0):
        days_until_sunday = 7
    next_sunday = (now + timedelta(days=days_until_sunday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (next_sunday - now).total_seconds()


def _run_weekly_change_check(job_id: str):
    """
    Per-flow change detection + org entity refresh.
    Stores flow_change chunks for any detected changes.
    Called by the weekly scheduler and the /refresh endpoint.
    """
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def update(msg):
        logger.info(msg)
        if job_id in _jobs:
            _jobs[job_id]["message"] = msg

    update("Fetching published flow list from Genesys...")
    try:
        all_flows = _fetch_all("/flows", {"publishedOnly": "true"})
    except Exception as e:
        update(f"Failed to fetch flows: {e}")
        if job_id in _jobs:
            _jobs[job_id].update({"status": "failed", "error": str(e)})
        _discord(f"**Genesys RAG** - Weekly refresh FAILED ({run_date})\nCould not fetch flow list: {e}")
        return

    candidate_flows = [
        f for f in all_flows
        if f.get("type", "").lower() in FULL_CONFIG_TYPES
        and f.get("active", True)
        and "healthcheck" not in f.get("name", "").lower()
        and "test"        not in f.get("name", "").lower()
        and "_bu"         not in f.get("name", "").lower()
        and "backup"      not in f.get("name", "").lower()
    ]
    update(f"Checking {len(candidate_flows)} flows for version changes...")

    col     = _get_collection()
    changes = []
    skipped = 0
    errors  = []

    for i, f in enumerate(candidate_flows):
        update(f"  [{i+1}/{len(candidate_flows)}] {f['name']}")
        try:
            result = run_flow_change_detection(f["id"], col)
            if result.get("changed"):
                change_chunk = _build_flow_change_chunk(result)
                _upsert(col, [change_chunk])

                new_chunks = result.get("new_chunks", [])
                if new_chunks:
                    _upsert(col, new_chunks)

                changes.append(result)
                logger.info("  Change recorded: %s (%d chunks updated)",
                            result["flow_name"], len(new_chunks))
            else:
                skipped += 1
        except Exception as e:
            logger.exception("Change detection failed for %s", f.get("name"))
            errors.append({"flow": f.get("name", f["id"]), "error": str(e)})

    update("Refreshing org entities (queues, policies, schedules, etc.)...")
    try:
        entity_summary = run_org_entities_refresh(status_dict=_jobs.get(job_id))
    except Exception as e:
        logger.exception("Org entity refresh failed")
        entity_summary = {"error": str(e)}

    if job_id in _jobs:
        _jobs[job_id].update({
            "status": "complete",
            "summary": {
                "flows_checked":  len(candidate_flows),
                "flows_changed":  len(changes),
                "flows_skipped":  skipped,
                "errors":         errors,
                "entity_refresh": entity_summary,
            },
        })

    for change in changes:
        _discord_change_report(change)

    if changes:
        _discord(
            f"**Genesys RAG** - Weekly refresh complete ({run_date})\n"
            f"{len(changes)} flow(s) changed | {skipped} unchanged | {len(errors)} error(s)\n"
            f"Change details stored in RAG - query for full diff."
        )
    else:
        _discord(
            f"**Genesys RAG** - Weekly refresh complete ({run_date})\n"
            f"No flow changes detected. {skipped} flows checked. Org entities refreshed."
        )

    if errors:
        error_lines = "\n".join(f"  - {e['flow']}: {e['error']}" for e in errors[:5])
        _discord(f"**Genesys RAG** - Refresh errors ({run_date}):\n{error_lines}")


def _weekly_refresh_loop():
    """Background thread: sleep until Sunday midnight, run change check, repeat."""
    while True:
        wait     = _seconds_until_next_sunday_midnight()
        next_run = datetime.now(timezone.utc) + timedelta(seconds=wait)
        logger.info(
            "Weekly refresh scheduled for %s UTC (%.1f hours from now)",
            next_run.strftime("%Y-%m-%d %H:%M"),
            wait / 3600,
        )
        threading.Event().wait(timeout=wait)

        job_id   = f"weekly_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Starting weekly change check (job %s)", job_id)
        _jobs[job_id] = {
            "job_id":       job_id,
            "flow_name":    "__all__",
            "status":       "running",
            "message":      "Weekly change detection started",
            "triggered_by": "scheduler",
        }
        _discord(f"**Genesys RAG** - Weekly change check started ({run_date})")

        try:
            _run_weekly_change_check(job_id)
        except Exception as e:
            logger.exception("Weekly change check crashed")
            _jobs[job_id].update({"status": "failed", "error": str(e)})
            _discord(f"**Genesys RAG** - Weekly refresh FAILED ({run_date})\nError: {e}")


@app.on_event("startup")
def start_scheduler():
    t = threading.Thread(target=_weekly_refresh_loop, daemon=True, name="weekly-refresh")
    t.start()
    logger.info("Weekly refresh scheduler started")


# -------------------------------------------------------------------------
# ChromaDB helper
# -------------------------------------------------------------------------

def _get_collection():
    os.environ.setdefault("CHROMA_CACHE_DIR", CHROMA_CACHE_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef     = embedding_functions.ONNXMiniLM_L6_V2()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:      str
    chunk_type: Optional[str] = None
    flow_name:  Optional[str] = None
    top_k:      int = 5


class IngestRequest(BaseModel):
    flow_name: str
    reset:     bool = False


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        col   = _get_collection()
        count = col.count()
        wait  = _seconds_until_next_sunday_midnight()
        next_refresh = (datetime.now(timezone.utc) + timedelta(seconds=wait)).strftime("%Y-%m-%d %H:%M UTC")
        return {
            "status":                 "ok",
            "collection_count":       count,
            "next_scheduled_refresh": next_refresh,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query")
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    col = _get_collection()

    filters = {}
    if req.chunk_type:
        filters["chunk_type"] = req.chunk_type
    if req.flow_name:
        filters["flow_name"] = req.flow_name

    where = None
    if len(filters) == 1:
        where = filters
    elif len(filters) > 1:
        where = {"$and": [{k: v} for k, v in filters.items()]}

    kwargs = {
        "query_texts": [req.query],
        "n_results":   req.top_k,
        "include":     ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = col.query(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    return {
        "query":   req.query,
        "results": [
            {
                "text":     doc,
                "score":    round(1 - dist, 4),
                "metadata": meta,
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ],
    }


def _run_ingest_job(job_id: str, flow_name: str, reset: bool):
    _jobs[job_id]["status"] = "running"
    try:
        if flow_name == "__all__":
            summary = run_org_ingest(reset=reset, status_dict=_jobs[job_id])
        else:
            summary = run_ingest(flow_name, reset=reset, status_dict=_jobs[job_id])
        _jobs[job_id]["status"]  = "complete"
        _jobs[job_id]["summary"] = summary
    except Exception as e:
        logger.exception("Ingest job %s failed", job_id)
        _jobs[job_id].update({"status": "failed", "error": str(e)})


@app.post("/ingest")
def start_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id":    job_id,
        "flow_name": req.flow_name,
        "status":    "queued",
        "message":   "",
    }
    background_tasks.add_task(_run_ingest_job, job_id, req.flow_name, req.reset)
    return {"status": "started", "job_id": job_id}


@app.get("/ingest/{job_id}")
def get_ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.post("/refresh")
def trigger_refresh(background_tasks: BackgroundTasks):
    """
    Manually trigger a change detection + org entity refresh.
    Same logic as the weekly job.
    """
    job_id = f"manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    _jobs[job_id] = {
        "job_id":       job_id,
        "flow_name":    "__all__",
        "status":       "running",
        "message":      "Manual change check triggered",
        "triggered_by": "api",
    }
    background_tasks.add_task(_run_weekly_change_check, job_id)
    return {"status": "started", "job_id": job_id}
