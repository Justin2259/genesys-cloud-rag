# Genesys Cloud RAG Pipeline

Semantic search over your entire Genesys Cloud org configuration. Ask plain-English questions about IVR routing, menus, ring group routing, ACD queues, schedules, recording policies, wrap-up codes, and anything else stored in Genesys Architect.

Built on ChromaDB + ONNX all-MiniLM-L6-v2 embeddings, deployed as a Docker container on a Linux VPS.

## Architecture

```
Local machine                          VPS (Docker: genesys-cloud-rag)
scripts/                               port 8765
  rag_ingest.py  --SSH tunnel-->       /ingest  (background job)
  rag_query.py   --HTTP-------->       /query   (semantic search)
                                       ChromaDB volume: /data/chroma_db
```

- API is bound to `127.0.0.1` only. Local scripts open an SSH tunnel automatically.
- ONNX model (~90MB) downloads on first ingest and caches on the host.

## Chunk Types

| chunk_type | What it contains |
|-----------|-----------------|
| `flow_overview` | Flow name, type, division, referenced queues/data tables |
| `menu_option` | One DTMF digit per chunk - digit, label, destination |
| `dynamic_group_routing` | Ring group routing via data table lookup |
| `task_flow` | Named sub-tasks and their action types |
| `data_table_reference` | Which data tables a flow uses, what columns they return |
| `queue` | ACD queue details: routing method, calling party, in-queue flow, wrap-up codes |
| `group` | Ring group / call group details |
| `data_table_schema` | Data table schema and sample rows |
| `prompt` | Architect prompt audio/text (batched) |
| `ivr_config` | IVR DNIS-to-flow mappings |
| `phone_number` | DID pool / assigned phone numbers |
| `schedule_group` | Open/closed/holiday schedule group with IANA timezone |
| `schedule` | Individual schedule definitions (batched) |
| `wrap_up_code` | One chunk per disposition code, with queue assignments |
| `flow_metadata` | Lightweight metadata for non-IVR flows |
| `recording_policy` | Media retention policy: which queues, retain/delete, retention days |
| `flow_change` | Change record: who published, when, what tasks/menus changed (before/after) |

## Setup

### 1. VPS: create credentials

```bash
mkdir /root/genesys-cloud-rag
cp .env.example /root/genesys-cloud-rag/.env
# Edit .env with your Genesys OAuth2 client credentials and region
```

### 2. VPS: deploy Docker container

```bash
scp Dockerfile requirements.txt ingest.py server.py root@your.vps.host.ip:/root/genesys-cloud-rag/

ssh root@your.vps.host.ip "
  cd /root/genesys-cloud-rag
  docker build -t genesys-cloud-rag:latest .
  docker run -d --name genesys-cloud-rag --restart unless-stopped \
    -p 127.0.0.1:8765:8765 \
    -v genesys-cloud-rag-data:/data \
    --env-file /root/genesys-cloud-rag/.env \
    genesys-cloud-rag:latest
"
```

### 3. Local scripts: configure SSH

Edit `VPS_HOST` and `SSH_KEY` at the top of both `scripts/rag_ingest.py` and `scripts/rag_query.py`.

### 4. First ingest

```bash
# Ingest a single flow
python scripts/rag_ingest.py --flow "My Main IVR"

# Or ingest everything at once (takes 60-90 minutes for a large org)
python scripts/rag_ingest.py --org
```

## Usage

### Ingest

```bash
# Single flow (most common - run after a flow is republished)
python scripts/rag_ingest.py --flow "My Main IVR"
python scripts/rag_ingest.py --flow "My Main IVR" --reset

# Full org ingest
python scripts/rag_ingest.py --org
python scripts/rag_ingest.py --org --reset

# Change detection + org entity refresh (same as weekly job)
python scripts/rag_ingest.py --refresh
```

### Query

```bash
python scripts/rag_query.py --query "what happens when caller presses 2"
python scripts/rag_query.py --query "voicemail routing" --chunk-type dynamic_group_routing
python scripts/rag_query.py --query "recording policy for my queue" --chunk-type recording_policy
python scripts/rag_query.py --query "what changed recently" --chunk-type flow_change
python scripts/rag_query.py --query "schedule timezone" --chunk-type schedule_group
python scripts/rag_query.py --query "wrap up codes for my queue" --chunk-type wrap_up_code
python scripts/rag_query.py --query "main menu options" --flow "My Main IVR" --verbose
```

## Weekly Change Detection

The server runs a weekly job every Sunday at 00:00 UTC:
1. Checks all published flows for version changes (compares stored vs current version ID)
2. For changed flows: diffs old vs new chunks (added/modified/removed tasks/menus)
3. Resolves who published the change from `publishedVersion.createdBy`
4. Stores a `flow_change` chunk in ChromaDB with full before/after diff
5. Re-ingests the updated flow chunks
6. Refreshes all org entities (queues, recording policies, schedules, etc.)
7. Posts per-flow change report to Discord (if webhook configured)

Trigger manually anytime:
```bash
python scripts/rag_ingest.py --refresh
```

## Customization

### Dynamic ring group routing detection

`ingest.py:_build_dynamic_group_chunks()` detects data table lookups that output ring group extension numbers. Update the `dynamic_group_patterns` set to match your org's data table output column names:

```python
dynamic_group_patterns = {
    "Live_Group_Extension", "VM_Group_Extension",
    # Add your column names here
}
```

### Backup flow filtering

Flows with `_bu` or `backup` in the name (case-insensitive) are excluded from full config ingest and change detection. They appear in `flow_metadata` chunks only.

## API Reference

All calls go through the SSH tunnel (handled automatically by the scripts).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | `{status, collection_count, next_scheduled_refresh}` |
| `/query` | POST | `{query, chunk_type?, flow_name?, top_k?}` -> `{results[]}` |
| `/ingest` | POST | `{flow_name, reset?}` -> `{status, job_id}` |
| `/ingest/{job_id}` | GET | `{status, message, summary?}` |
| `/refresh` | POST | Trigger change detection + org refresh -> `{status, job_id}` |

## Troubleshooting

**API not reachable:**
```bash
ssh root@your.vps.host.ip "docker ps --filter name=genesys-cloud-rag"
ssh root@your.vps.host.ip "docker logs genesys-cloud-rag --tail 50"
ssh root@your.vps.host.ip "docker restart genesys-cloud-rag"
```

**ChromaDB HNSW desync (container hangs on startup after a hard kill during ingest):**
```bash
ssh root@your.vps.host.ip "
  docker exec genesys-cloud-rag python3 -c \"
import sqlite3, os
db = os.path.join(os.environ['CHROMA_DB_PATH'], 'chroma.sqlite3')
conn = sqlite3.connect(db)
conn.execute('DELETE FROM embeddings_queue')
conn.commit()
conn.close()
print('Queue cleared')
\"
"
```
Then restart the container.

**VPS memory pressure during ingest:**
The ONNX model needs at least 1GB free RAM. Check with: `ssh ... "free -h"`

**ONNX model cache location:**
The model caches at `/root/.cache/chroma/onnx_models/` on the container host (not inside the `/data/` volume).
