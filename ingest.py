#!/usr/bin/env python3
"""
Genesys Cloud RAG ingest module.
Parses Genesys flow configs and org entities into searchable ChromaDB chunks.

Chunk types produced:
  flow_overview            - Top-level flow metadata and referenced entities
  menu_option              - Individual DTMF digit options in a menu
  dynamic_group_routing    - Ring group routing via data table lookup
  data_table_reference     - Data table usage within a flow
  task_flow                - Named sub-task within a flow
  queue                    - ACD queue details (routing method, calling party, in-queue flow)
  group                    - Ring group / call group details
  data_table_schema        - Data table schema and sample rows
  prompt                   - Architect prompts (batched)
  ivr_config               - IVR DNIS-to-flow mappings
  phone_number             - DID pool / assigned phone numbers
  schedule_group           - Open/closed/holiday schedule group with timezone
  schedule                 - Individual schedule definitions (batched)
  wrap_up_code             - Wrap-up (disposition) codes with queue assignments
  flow_metadata            - Lightweight metadata for non-IVR flows
  recording_policy         - Media retention policy (conditions + actions)
  flow_change              - Change record: who published, when, what changed
"""
import os
import re
import time
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

GENESYS_CLIENT_ID     = os.getenv("GENESYS_CLIENT_ID")
GENESYS_CLIENT_SECRET = os.getenv("GENESYS_CLIENT_SECRET")
GENESYS_REGION        = os.getenv("GENESYS_REGION", "mypurecloud.com")
CHROMA_DB_PATH        = os.getenv("CHROMA_DB_PATH", "/data/chroma_db")
CHROMA_CACHE_DIR      = os.getenv("CHROMA_CACHE_DIR", "/data/.cache/chroma")
COLLECTION_NAME       = "genesys_ivr"

REGION_AUTH = {
    "mypurecloud.com":        "https://login.mypurecloud.com",
    "usw2.pure.cloud":        "https://login.usw2.pure.cloud",
    "use2.us-gov-pure.cloud": "https://login.use2.us-gov-pure.cloud",
    "cac1.pure.cloud":        "https://login.cac1.pure.cloud",
    "mypurecloud.ie":         "https://login.mypurecloud.ie",
    "euw2.pure.cloud":        "https://login.euw2.pure.cloud",
    "mypurecloud.de":         "https://login.mypurecloud.de",
    "aps1.pure.cloud":        "https://login.aps1.pure.cloud",
    "mypurecloud.jp":         "https://login.mypurecloud.jp",
    "apne2.pure.cloud":       "https://login.apne2.pure.cloud",
    "mypurecloud.com.au":     "https://login.mypurecloud.com.au",
    "sae1.pure.cloud":        "https://login.sae1.pure.cloud",
    "mec1.pure.cloud":        "https://login.mec1.pure.cloud",
    "apne3.pure.cloud":       "https://login.apne3.pure.cloud",
}

_token_cache: dict = {}
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# -------------------------------------------------------------------------
# Auth + API helpers
# -------------------------------------------------------------------------

def _get_token() -> str:
    global _token_cache
    now = time.time()
    if _token_cache.get("expires_at", 0) > now + 60:
        return _token_cache["access_token"]
    auth_url = REGION_AUTH.get(GENESYS_REGION)
    if not auth_url:
        raise ValueError(f"Unknown region: {GENESYS_REGION}")
    resp = requests.post(
        f"{auth_url}/oauth/token",
        data={"grant_type": "client_credentials"},
        auth=(GENESYS_CLIENT_ID, GENESYS_CLIENT_SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache = {
        "access_token": data["access_token"],
        "expires_at": now + data.get("expires_in", 3600),
    }
    return _token_cache["access_token"]


def _base() -> str:
    return f"https://api.{GENESYS_REGION}/api/v2"


def _get(path: str, params: dict = None) -> dict:
    token = _get_token()
    resp = requests.get(
        f"{_base()}{path}",
        params=params,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=60,
    )
    if resp.status_code in (403, 404):
        return {"_error": f"{resp.status_code}"}
    resp.raise_for_status()
    return resp.json()


def _fetch_all(path: str, extra: dict = None, max_pages: int = 50) -> list:
    """Paginate through all pages of a Genesys list endpoint."""
    params = {"pageSize": 100, "pageNumber": 1}
    if extra:
        params.update(extra)
    items = []
    for page in range(1, max_pages + 1):
        params["pageNumber"] = page
        d = _get(path, params)
        if "_error" in d:
            logger.warning("Fetch error on %s page %d: %s", path, page, d["_error"])
            break
        if isinstance(d, list):
            items.extend(d)
            break
        page_items = d.get("entities", d.get("items", d.get("data", [])))
        items.extend(page_items)
        if page >= d.get("pageCount", 1):
            break
        time.sleep(0.05)
    return items


# -------------------------------------------------------------------------
# ChromaDB
# -------------------------------------------------------------------------

def _get_collection():
    os.environ.setdefault("CHROMA_CACHE_DIR", CHROMA_CACHE_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.ONNXMiniLM_L6_V2()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _upsert(collection, chunks: list[dict]):
    if not chunks:
        return
    BATCH = 50
    ids   = [c["chunk_id"] for c in chunks]
    docs  = [c["text"] for c in chunks]
    metas = [
        {k: v for k, v in c.items()
         if k not in ("chunk_id", "text") and isinstance(v, (str, int, float, bool))}
        for c in chunks
    ]
    for i in range(0, len(chunks), BATCH):
        collection.upsert(
            ids=ids[i:i+BATCH],
            documents=docs[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
        )


def _safe_id(*parts: str) -> str:
    raw = "__".join(str(p) for p in parts)
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)[:512]


# =========================================================================
# FLOW CONFIG PARSING (individual flow ingest)
# =========================================================================

def _fetch_flow_config(flow_name: str) -> dict:
    if _UUID_RE.match(flow_name.strip()):
        flow_id = flow_name.strip()
        meta = _get(f"/flows/{flow_id}")
        flow = {
            "id": flow_id,
            "name": meta.get("name", flow_id),
            "type": meta.get("type", "inboundcall"),
            "division": meta.get("division", {}),
            "description": meta.get("description", ""),
        }
    else:
        encoded = requests.utils.quote(flow_name.strip())
        search = _get(f"/flows?name={encoded}&pageSize=25")
        flows = search.get("entities", [])
        if not flows:
            raise ValueError(f"No flow found matching '{flow_name}'")
        exact = [f for f in flows if f["name"].strip().lower() == flow_name.strip().lower()]
        flow = exact[0] if exact else flows[0]

    flow_id = flow["id"]
    logger.info("Fetching config for flow: %s (%s)", flow["name"], flow_id)
    full_meta = _get(f"/flows/{flow_id}")
    pub = full_meta.get("publishedVersion") or {}
    published_version_id = pub.get("id", "")
    published_date       = pub.get("datePublished", "")
    created_by_id        = (pub.get("createdBy") or {}).get("id", "")

    try:
        cfg = _get(f"/flows/{flow_id}/latestconfiguration")
    except Exception:
        if not published_version_id:
            raise ValueError(f"No published version for flow {flow_id}")
        cfg = _get(f"/flows/{flow_id}/versions/{published_version_id}/configuration")

    return {
        "flow_id":              flow_id,
        "flow_name":            flow["name"],
        "flow_type":            flow.get("type", "inboundcall"),
        "division":             flow.get("division", {}).get("name", "") if isinstance(flow.get("division"), dict) else "",
        "description":          flow.get("description", ""),
        "config":               cfg,
        "published_version_id": published_version_id,
        "published_date":       published_date,
        "published_by_id":      created_by_id,
    }


def _walk_actions(obj, visited=None):
    if visited is None:
        visited = set()
    if isinstance(obj, dict):
        obj_id = obj.get("id")
        if obj_id:
            if obj_id in visited:
                return
            visited.add(obj_id)
        if "__type" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_actions(v, visited)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_actions(item, visited)


def _build_flow_overview(flow: dict) -> dict:
    cfg = flow["config"]
    text = (
        f"Flow: {flow['flow_name']}\n"
        f"Type: {flow['flow_type']}\n"
        f"Division: {flow['division']}\n"
        f"Description: {flow['description'] or 'None'}\n"
        f"Default language: {cfg.get('defaultLanguage', 'en-US')}\n"
    )
    manifest = cfg.get("manifest", {})
    for key in ["queue", "dataTable", "scheduleGroup", "flow"]:
        items = manifest.get(key, [])
        if items:
            names = [i.get("name", i.get("id", "?")) for i in items]
            text += f"{key.capitalize()} refs: {', '.join(names)}\n"
    chunk = {
        "chunk_id":   _safe_id(flow["flow_id"], "overview"),
        "chunk_type": "flow_overview",
        "flow_name":  flow["flow_name"],
        "flow_id":    flow["flow_id"],
        "text":       text.strip(),
    }
    if flow.get("published_version_id"):
        chunk["published_version_id"] = flow["published_version_id"]
    if flow.get("published_date"):
        chunk["published_date"] = flow["published_date"]
    if flow.get("published_by_id"):
        chunk["published_by_id"] = flow["published_by_id"]
    return chunk


def _build_menu_option_chunks(flow: dict) -> list[dict]:
    cfg = flow["config"]
    chunks = []
    for seq in cfg.get("flowSequenceItemList", []):
        if not seq.get("menuChoiceList"):
            continue
        menu_name = seq.get("name", "Unknown Menu")
        prompts = seq.get("prompts", [])
        prompt_text = next(
            (p.get("tts", "").strip() for p in prompts if isinstance(p, dict) and p.get("tts", "").strip()),
            ""
        )
        for choice in seq.get("menuChoiceList", []):
            digit  = choice.get("digit")
            name   = choice.get("name", "Unknown")
            action = choice.get("action", {})
            text = (
                f"Flow: {flow['flow_name']}\n"
                f"Menu: {menu_name}\n"
                f"Digit: {digit if digit is not None else '(speech)'}\n"
                f"Option: {name}\n"
            )
            if prompt_text:
                text += f"Prompt: {prompt_text}\n"
            dest = action.get("name", "")
            if dest:
                text += f"Routes to: {dest} ({action.get('__type', '')})\n"
            chunks.append({
                "chunk_id": _safe_id(flow["flow_id"], "menu", menu_name, str(digit), name),
                "chunk_type": "menu_option",
                "flow_name": flow["flow_name"],
                "flow_id": flow["flow_id"],
                "menu_name": menu_name,
                "digit": str(digit) if digit is not None else "speech",
                "text": text.strip(),
            })
    return chunks


def _build_data_table_chunks(flow: dict) -> list[dict]:
    cfg  = flow["config"]
    seen: dict = {}
    for action in _walk_actions(cfg):
        if action.get("__type") != "DataTableLookupAction":
            continue
        tname = action.get("datatableName", "Unknown")
        tid   = action.get("datatableId", "")
        if tname in seen:
            seen[tname]["lookup_count"] += 1
            continue
        outputs = action.get("outputs", [])
        cols    = [o.get("name", "?") for o in outputs if isinstance(o, dict)]
        key_exp = action.get("lookupKeyValue", {}).get("text", "")
        seen[tname] = {"table_id": tid, "output_columns": cols,
                       "key_expression": key_exp, "lookup_count": 1}
    chunks = []
    for tname, info in seen.items():
        text = (
            f"Flow: {flow['flow_name']}\n"
            f"Data table lookup: {tname}\n"
            f"Table ID: {info['table_id']}\n"
            f"Lookup key: {info['key_expression']}\n"
            f"Output columns: {', '.join(info['output_columns'])}\n"
            f"Used {info['lookup_count']} time(s) in flow\n"
        )
        chunks.append({
            "chunk_id": _safe_id(flow["flow_id"], "datatable", tname),
            "chunk_type": "data_table_reference",
            "flow_name": flow["flow_name"],
            "flow_id": flow["flow_id"],
            "table_name": tname,
            "text": text.strip(),
        })
    return chunks


def _build_dynamic_group_chunks(flow: dict) -> list[dict]:
    """
    Detect DataTableLookupAction nodes that output ring group extension columns
    and build a human-readable routing description chunk.

    Customize `dynamic_group_patterns` to match the output column names from
    your data table that hold ring group extension values.
    """
    cfg = flow["config"]

    # Customize these to match the output column names in your data table
    # that contain ring group extension numbers.
    dynamic_group_patterns = {
        "Live_Group_Extension", "VM_Group_Extension",
        # Add your org-specific column name variants here:
        # "LiveGroupExtension", "VMGroupExtension",
    }

    seq_by_action: dict = {}
    for seq in cfg.get("flowSequenceItemList", []):
        sname = seq.get("name", "Unknown")
        for action in _walk_actions(seq.get("actionList", [])):
            aid = action.get("id")
            if aid:
                seq_by_action[aid] = sname

    chunks = []
    seen: set = set()
    for action in _walk_actions(cfg):
        if action.get("__type") != "DataTableLookupAction":
            continue
        outputs   = action.get("outputs", [])
        col_names = {o.get("name", "") for o in outputs if isinstance(o, dict)}
        if not (col_names & dynamic_group_patterns):
            continue
        tname      = action.get("datatableName", "Unknown")
        action_id  = action.get("id", "")
        ctx        = seq_by_action.get(action_id, "Main IVR")
        key        = f"{tname}__{ctx}"
        if key in seen:
            continue
        seen.add(key)

        col_to_var = {
            o.get("name", ""): o.get("value", {}).get("text", "")
            for o in outputs if isinstance(o, dict)
        }
        vm_col   = next((c for c in col_names if "VM" in c or "Vm" in c), "")
        live_col = next((c for c in col_names
                         if "Group" in c and "VM" not in c and "Vm" not in c), "")

        text = (
            f"Flow: {flow['flow_name']}\n"
            f"Context: {ctx}\n"
            f"Routing type: Dynamic ring group routing\n"
            f"Data source: {tname} (DataTableLookupAction)\n\n"
            f"How it works:\n"
            f"  The flow looks up ring group extension(s) from {tname}\n"
            f"  using the DNIS (phone number called) as the lookup key.\n\n"
            f"  Voicemail ring group column: '{vm_col}'\n"
            f"    Variable: {col_to_var.get(vm_col, '')}\n"
            f"    Behavior: Caller hears voicemail greeting, leaves message\n\n"
            f"  Live call ring group column: '{live_col}'\n"
            f"    Variable: {col_to_var.get(live_col, '')}\n"
            f"    Behavior: Live agents answer the call\n\n"
            f"  Voicemail path: VM ring group -> overflow -> live ring group\n"
            f"  This is NOT an ACD queue. Direct ring group transfer (FindGroup pattern).\n"
        )
        chunks.append({
            "chunk_id": _safe_id(flow["flow_id"], "dynrouting", key),
            "chunk_type": "dynamic_group_routing",
            "flow_name": flow["flow_name"],
            "flow_id": flow["flow_id"],
            "data_table": tname,
            "context": ctx,
            "text": text.strip(),
        })
    return chunks


def _build_task_flow_chunks(flow: dict) -> list[dict]:
    cfg      = flow["config"]
    flow_type = flow.get("flow_type", "").upper()
    chunks   = []

    HIGHLIGHT_ACTIONS = {
        "HoldMusicAction":            "Hold music played",
        "OfferCallbackAction":        "Callback offered to caller",
        "CallBackAction":             "Callback offered to caller",
        "AskForSlotAction":           "Bot collects slot/input from caller",
        "ExitBotFlowAction":          "Bot session exits",
        "EvaluateScheduleGroupAction":"Schedule group evaluated (open/closed/holiday)",
        "EvaluateScheduleAction":     "Schedule evaluated (open/closed)",
        "SendResponseAction":         "Response sent (SMS/chat)",
        "GetResponseAction":          "Customer input collected (SMS/chat)",
        "PlayAudioAction":            "Audio prompt played",
        "CollectInputAction":         "DTMF/speech input collected from caller",
        "DisconnectAction":           "Call/session disconnected",
        "ScreenPopAction":            "Screen pop triggered for agent",
        "SetExternalTagAction":       "External tag set on interaction",
        "CallCommonModuleAction":     "Common module invoked",
        "DataAction":                 "External data action called",
        "WaitAction":                 "Flow waits (callback/async)",
        "ClearVoicemailSnippetAction":"Voicemail snippet cleared",
        "FindUserPromptAction":       "Dynamic user prompt looked up",
    }

    TASK_KIND_MAP = {
        "Task":             "Task",
        "CommonModuleTask": "Common module task",
        "LoopTask":         "Loop task",
        "BotState":         "Bot state",
        "State":            "State",
        "Menu":             "Menu",
    }

    TOP_LEVEL_KEYS = {
        "inboundcall":          "taskList",
        "commonmodule":         "taskList",
        "inqueuecall":          "loopTaskList",
        "bot":                  "stateList",
        "digitalbot":           "stateList",
        "inboundshortmessage":  "stateList",
        "inboundchat":          "stateList",
        "outboundcall":         "menuList",
        "inqueueshortmessage":  "loopTaskList",
        "voicemail":            "taskList",
        "workflow":             "stateList",
    }

    top_key  = TOP_LEVEL_KEYS.get(flow.get("flow_type", "").lower(), "taskList")
    task_list = cfg.get(top_key, [])

    for task in task_list:
        tname     = task.get("name", "Unnamed")
        task_type = task.get("__type", "Task")
        kind      = TASK_KIND_MAP.get(task_type, task_type)

        action_types = []
        transfer_refs = []
        data_actions  = []

        for action in _walk_actions(task.get("actionList", task.get("actions", []))):
            atype = action.get("__type", "")
            if atype:
                action_types.append(atype)
            if atype in ("TransferToAcdAction", "TransferToQueueAction"):
                q = action.get("queue", action.get("queueExpression", {}))
                if isinstance(q, dict):
                    qname = q.get("name", q.get("text", ""))
                    if qname:
                        transfer_refs.append(qname)
            if atype == "DataAction":
                da = action.get("dataAction", {})
                if isinstance(da, dict):
                    da_name = da.get("name", "")
                    if da_name:
                        data_actions.append(da_name)

        unique_types = list(dict.fromkeys(action_types))

        text = (
            f"Flow: {flow['flow_name']}\n"
            f"Flow type: {flow_type}\n"
            f"Task: {tname}\n"
            f"Task kind: {kind}\n"
            f"Action types: {', '.join(unique_types) or 'none'}\n"
        )
        if transfer_refs:
            text += f"Transfers to queue(s): {', '.join(transfer_refs)}\n"
        if data_actions:
            text += f"Data actions called: {', '.join(data_actions)}\n"
        for atype, label in HIGHLIGHT_ACTIONS.items():
            if atype in action_types:
                text += f"Notable: {label}\n"

        chunks.append({
            "chunk_id": _safe_id(flow["flow_id"], "task", tname),
            "chunk_type": "task_flow",
            "flow_name": flow["flow_name"],
            "flow_id": flow["flow_id"],
            "task_name": tname,
            "text": text.strip(),
        })
    return chunks


# =========================================================================
# ORG-WIDE ENTITY CHUNK BUILDERS
# =========================================================================

def _fetch_queue_wrapup_map(queue_ids: list[str]) -> dict:
    """
    Fetch wrapup codes assigned to each queue.
    Returns {queue_id: [wrapup_code_dict, ...]}
    """
    queue_to_wrapups: dict = {}
    for qid in queue_ids:
        try:
            codes = _fetch_all(f"/routing/queues/{qid}/wrapupcodes")
            queue_to_wrapups[qid] = codes
        except Exception as e:
            logger.debug("Could not fetch wrapup codes for queue %s: %s", qid, e)
    return queue_to_wrapups


def _chunks_queues(queues: list, queue_wrapup_map: dict = None) -> list[dict]:
    qwmap = queue_wrapup_map or {}
    chunks = []
    for q in queues:
        name     = q.get("name", "Unknown")
        qid      = q.get("id", "")
        desc     = q.get("description", "")
        division = q.get("division", {}).get("name", "") if isinstance(q.get("division"), dict) else ""
        media    = q.get("mediaSettings", {})
        members  = q.get("memberCount", q.get("memberDetailCount", "?"))
        acw      = q.get("acwSettings", {}).get("wrapupPrompt", "")
        routing_method  = q.get("skillEvaluationMethod", "")
        calling_name    = q.get("callingPartyName", "")
        calling_number  = q.get("callingPartyNumber", "")
        queue_flow_ref  = q.get("queueFlow", {})
        queue_flow_name = queue_flow_ref.get("name", "") if isinstance(queue_flow_ref, dict) else ""
        call_settings    = media.get("call", {}) if isinstance(media, dict) else {}
        alerting_timeout = call_settings.get("alertingTimeoutSeconds", "")
        auto_answer      = call_settings.get("autoAnswerAlertToneSeconds", "")
        wrapup_codes     = [c.get("name", "") for c in qwmap.get(qid, []) if c.get("name")]

        text = (
            f"Queue: {name}\n"
            f"ID: {qid}\n"
            f"Division: {division}\n"
            f"Description: {desc or 'None'}\n"
            f"Member count: {members}\n"
            f"Routing method: {routing_method or 'Not specified'}\n"
            f"ACW/wrap-up prompt: {acw or 'None'}\n"
        )
        if calling_name or calling_number:
            text += f"Calling party: {calling_name or 'N/A'} / {calling_number or 'N/A'}\n"
        if queue_flow_name:
            text += f"In-queue flow: {queue_flow_name}\n"
        if alerting_timeout:
            text += f"Alerting timeout (calls): {alerting_timeout} seconds\n"
        if auto_answer:
            text += f"Auto-answer tone seconds: {auto_answer}\n"
        if wrapup_codes:
            text += f"Wrap-up codes ({len(wrapup_codes)}): {', '.join(wrapup_codes)}\n"

        chunks.append({
            "chunk_id": _safe_id("queue", qid),
            "chunk_type": "queue",
            "entity_name": name,
            "entity_id": qid,
            "text": text.strip(),
        })
    return chunks


def _chunks_groups(groups: list) -> list[dict]:
    chunks = []
    for g in groups:
        name    = g.get("name", "Unknown")
        gid     = g.get("id", "")
        desc    = g.get("description", "")
        gtype   = g.get("type", "")
        members = g.get("memberCount", "?")
        # Detect voicemail vs live ring group from naming convention.
        # Customize these patterns to match your org's ring group naming.
        is_vm   = "voicemail" in name.lower() or name.strip().endswith("- Voicemail")
        ring_note = ""
        if is_vm:
            ring_note = "Role: Voicemail ring group\n"

        text = (
            f"Group: {name}\n"
            f"ID: {gid}\n"
            f"Type: {gtype or 'ring group'}\n"
            f"Description: {desc or 'None'}\n"
            f"Member count: {members}\n"
            + ring_note
        )
        chunks.append({
            "chunk_id": _safe_id("group", gid),
            "chunk_type": "group",
            "entity_name": name,
            "entity_id": gid,
            "text": text.strip(),
        })
    return chunks


def _chunks_datatables(tables: list) -> list[dict]:
    """Schema + sample rows for each data table."""
    chunks = []
    for t in tables:
        name   = t.get("name", "Unknown")
        tid    = t.get("id", "")
        schema = t.get("schema", {})
        props  = schema.get("properties", {})

        cols = []
        for col_name, col_def in props.items():
            col_type = col_def.get("type", "string")
            is_key   = col_def.get("$key", False)
            cols.append(f"  {col_name} ({col_type})" + (" [KEY]" if is_key else ""))

        rows_data = _get(f"/flows/datatables/{tid}/rows?pageSize=10&showbrief=false")
        sample_rows = []
        if not rows_data.get("_error"):
            entities = rows_data.get("entities", [])
            for row in entities[:5]:
                row_vals = {k: v for k, v in row.items() if not k.startswith("_")}
                sample_rows.append(json.dumps(row_vals))

        total_rows = rows_data.get("total", rows_data.get("count", "?"))

        text = (
            f"Data table: {name}\n"
            f"ID: {tid}\n"
            f"Total rows: {total_rows}\n"
            f"Columns:\n" + "\n".join(cols) + "\n"
        )
        if sample_rows:
            text += f"\nSample rows (first {len(sample_rows)}):\n"
            for r in sample_rows:
                text += f"  {r}\n"

        chunks.append({
            "chunk_id": _safe_id("datatable_schema", tid),
            "chunk_type": "data_table_schema",
            "entity_name": name,
            "entity_id": tid,
            "text": text.strip(),
        })
        time.sleep(0.1)
    return chunks


def _chunks_prompts(prompts: list) -> list[dict]:
    chunks = []
    batch_text = ""
    batch_size = 0
    batch_num  = 0

    for p in prompts:
        name  = p.get("name", "Unknown")
        desc  = p.get("description", "")
        langs = [r.get("language", "") for r in p.get("resources", []) if isinstance(r, dict)]
        batch_text += f"  {name}"
        if desc:
            batch_text += f" - {desc}"
        if langs:
            batch_text += f" [{', '.join(langs)}]"
        batch_text += "\n"
        batch_size += 1

        if batch_size >= 50:
            batch_num += 1
            chunks.append({
                "chunk_id": _safe_id("prompts_batch", str(batch_num)),
                "chunk_type": "prompt",
                "entity_name": f"Prompts batch {batch_num}",
                "text": f"Architect prompts (batch {batch_num}, {batch_size} prompts):\n" + batch_text.strip(),
            })
            batch_text = ""
            batch_size = 0

    if batch_size > 0:
        batch_num += 1
        chunks.append({
            "chunk_id": _safe_id("prompts_batch", str(batch_num)),
            "chunk_type": "prompt",
            "entity_name": f"Prompts batch {batch_num}",
            "text": f"Architect prompts (batch {batch_num}, {batch_size} prompts):\n" + batch_text.strip(),
        })
    return chunks


def _chunks_ivrs(ivrs: list) -> list[dict]:
    chunks = []
    for ivr in ivrs:
        name     = ivr.get("name", "Unknown")
        iid      = ivr.get("id", "")
        dnis     = ivr.get("dnis", [])
        flow_ref = ivr.get("openHoursFlow", ivr.get("flow", {}))
        flow_name = flow_ref.get("name", "") if isinstance(flow_ref, dict) else ""
        sched_grp = ivr.get("scheduleGroup", {})
        sched_name = sched_grp.get("name", "") if isinstance(sched_grp, dict) else ""

        text = (
            f"IVR Config: {name}\n"
            f"ID: {iid}\n"
            f"Assigned flow: {flow_name or 'None'}\n"
            f"Schedule group: {sched_name or 'None'}\n"
            f"DNIS numbers ({len(dnis)}): {', '.join(str(d) for d in dnis[:20])}"
            + ("..." if len(dnis) > 20 else "") + "\n"
        )
        chunks.append({
            "chunk_id": _safe_id("ivr", iid),
            "chunk_type": "ivr_config",
            "entity_name": name,
            "entity_id": iid,
            "flow_name": flow_name,
            "text": text.strip(),
        })
    return chunks


def _chunks_did_pools(pools: list) -> list[dict]:
    chunks = []
    for pool in pools:
        name   = pool.get("name", "Unknown")
        pid    = pool.get("id", "")
        start  = pool.get("startPhoneNumber", "")
        end    = pool.get("endPhoneNumber", "")
        desc   = pool.get("description", "")
        region = pool.get("countryCode", "")
        text = (
            f"DID Pool: {name}\n"
            f"ID: {pid}\n"
            f"Range: {start} - {end}\n"
            f"Country: {region}\n"
            f"Description: {desc or 'None'}\n"
        )
        chunks.append({
            "chunk_id": _safe_id("didpool", pid),
            "chunk_type": "phone_number",
            "entity_name": name,
            "entity_id": pid,
            "text": text.strip(),
        })
    return chunks


def _chunks_schedule_groups(sgs: list) -> list[dict]:
    chunks = []
    for sg in sgs:
        name    = sg.get("name", "Unknown")
        sgid    = sg.get("id", "")
        div     = sg.get("division", {}).get("name", "") if isinstance(sg.get("division"), dict) else ""
        tz      = sg.get("timeZone", "")  # IANA timezone, e.g. America/Chicago
        open_s  = sg.get("openSchedules", [])
        closed  = sg.get("closedSchedules", [])
        holiday = sg.get("holidaySchedules", [])

        open_names    = [s.get("name", "") for s in open_s]
        closed_names  = [s.get("name", "") for s in closed]
        holiday_names = [s.get("name", "") for s in holiday]

        text = (
            f"Schedule group: {name}\n"
            f"ID: {sgid}\n"
            f"Division: {div}\n"
            f"Timezone: {tz or 'Not specified'}\n"
            f"Open schedules: {', '.join(open_names) or 'None'}\n"
            f"Closed schedules: {', '.join(closed_names) or 'None'}\n"
            f"Holiday schedules: {', '.join(holiday_names) or 'None'}\n"
        )
        chunks.append({
            "chunk_id": _safe_id("schedgrp", sgid),
            "chunk_type": "schedule_group",
            "entity_name": name,
            "entity_id": sgid,
            "timezone": tz,
            "text": text.strip(),
        })
    return chunks


def _chunks_schedules(schedules: list) -> list[dict]:
    chunks  = []
    batches = [schedules[i:i+30] for i in range(0, len(schedules), 30)]
    for b_idx, batch in enumerate(batches):
        lines = []
        for s in batch:
            name  = s.get("name", "Unknown")
            start = s.get("start", "")
            end   = s.get("end", "")
            rrule = s.get("rrule", "")
            lines.append(f"  {name} | {start} to {end} | {rrule}")
        text = f"Schedules (batch {b_idx+1}, {len(batch)} entries):\n" + "\n".join(lines)
        chunks.append({
            "chunk_id": _safe_id("schedules_batch", str(b_idx)),
            "chunk_type": "schedule",
            "entity_name": f"Schedules batch {b_idx+1}",
            "text": text.strip(),
        })
    return chunks


def _chunks_wrapup_codes(codes: list, wrapup_to_queues: dict = None) -> list[dict]:
    """One chunk per wrap-up code for high-fidelity semantic retrieval."""
    if not codes:
        return []
    wqmap = wrapup_to_queues or {}
    chunks = []
    for c in codes:
        code_id   = c.get("id", "")
        name      = c.get("name", "Unknown")
        div       = c.get("division", {}).get("name", "") if isinstance(c.get("division"), dict) else ""
        queues_using = wqmap.get(code_id, [])
        text = (
            f"Wrap-up code: {name}\n"
            f"Also known as: disposition code, ACW code, after-call work code.\n"
            f"Agents select this code after a call to record the outcome or call type.\n"
            f"Division: {div or 'N/A'}\n"
            f"ID: {code_id}\n"
        )
        if queues_using:
            text += f"Assigned to queues ({len(queues_using)}): {', '.join(queues_using)}\n"
        chunks.append({
            "chunk_id": _safe_id("wrapup", code_id or name),
            "chunk_type": "wrap_up_code",
            "entity_name": name,
            "text": text.strip(),
        })
    return chunks


def _chunks_recording_policies(policies: list, queue_name_map: dict = None, wrapup_name_map: dict = None) -> list[dict]:
    """One chunk per media retention / recording policy."""
    chunks = []
    for p in policies:
        pid     = p.get("id", "")
        name    = p.get("name", "Unknown")
        enabled = p.get("enabled", False)
        deleted = p.get("deleted", False)
        if deleted:
            continue

        media_policies = p.get("mediaPolicies", {})
        call_policy    = media_policies.get("callPolicy", {})
        conditions     = call_policy.get("conditions", {})
        actions        = call_policy.get("actions", {})

        queue_refs  = conditions.get("forQueues", [])
        wrapup_refs = conditions.get("wrapupCodes", [])
        directions  = conditions.get("directions", [])
        date_ranges = conditions.get("dateRanges", [])
        time_allowed = conditions.get("timeAllowed", {})

        # Genesys nested refs only carry id/selfUri - resolve names from provided maps
        qmap = queue_name_map or {}
        queue_names  = [
            qmap.get(q.get("id", ""), q.get("name") or q.get("id", "?"))
            for q in queue_refs if isinstance(q, dict)
        ]
        wmap = wrapup_name_map or {}
        wrapup_names = [
            wmap.get(w.get("id", ""), w.get("name") or w.get("id", "?"))
            for w in wrapup_refs if isinstance(w, dict)
        ]

        retain_recording = actions.get("retainRecording", False)
        delete_recording = actions.get("deleteRecording", False)
        retention_dur    = actions.get("retentionDuration", {})
        retention_days   = retention_dur.get("days", 0) if isinstance(retention_dur, dict) else 0
        assign_evals     = actions.get("assignEvaluations", [])
        assign_metered   = actions.get("assignMeteredAssignmentByAgent", [])

        text = (
            f"Recording policy: {name}\n"
            f"Enabled: {enabled}\n"
        )
        if queue_names:
            text += f"Applies to queues: {', '.join(queue_names)}\n"
        else:
            text += "Applies to queues: All queues (no queue filter - catch-all)\n"
        if wrapup_names:
            text += f"Applies to wrap-up codes: {', '.join(wrapup_names)}\n"
        if directions:
            text += f"Call directions: {', '.join(str(d) for d in directions)}\n"
        if date_ranges:
            text += f"Date ranges: {', '.join(str(d) for d in date_ranges)}\n"
        if time_allowed:
            time_zone = time_allowed.get("timeZone", {})
            tz_id = time_zone.get("id", "") if isinstance(time_zone, dict) else str(time_zone)
            time_slots = time_allowed.get("timeSlots", [])
            if tz_id:
                text += f"Time allowed timezone: {tz_id}\n"
            if time_slots:
                text += f"Time slots: {len(time_slots)} defined\n"
        text += f"Action - Retain recording: {retain_recording}\n"
        text += f"Action - Delete recording: {delete_recording}\n"
        if retention_days:
            text += f"Retention duration: {retention_days} days\n"
        if assign_evals:
            text += f"Assigns evaluations: {len(assign_evals)} form(s)\n"
        if assign_metered:
            text += f"Assigns metered assignment by agent: yes\n"
        text += f"ID: {pid}\n"

        chunks.append({
            "chunk_id": _safe_id("recpolicy", pid),
            "chunk_type": "recording_policy",
            "entity_name": name,
            "entity_id": pid,
            "enabled": enabled,
            "text": text.strip(),
        })
    return chunks


def _chunks_flow_metadata(flows: list, skip_ids: set) -> list[dict]:
    """Lightweight metadata chunks for flows not given full config treatment."""
    chunks = []
    for f in flows:
        fid   = f.get("id", "")
        if fid in skip_ids:
            continue
        name  = f.get("name", "Unknown")
        ftype = f.get("type", "")
        div   = f.get("division", {}).get("name", "") if isinstance(f.get("division"), dict) else ""
        desc  = f.get("description", "")
        text  = (
            f"Flow: {name}\n"
            f"Type: {ftype}\n"
            f"Division: {div}\n"
            f"Description: {desc or 'None'}\n"
            f"ID: {fid}\n"
        )
        chunks.append({
            "chunk_id": _safe_id("flow_meta", fid),
            "chunk_type": "flow_metadata",
            "flow_name": name,
            "flow_id": fid,
            "flow_type": ftype,
            "text": text.strip(),
        })
    return chunks


# =========================================================================
# CHANGE DETECTION
# =========================================================================

def _build_flow_chunks(flow: dict) -> list[dict]:
    """Build all chunks for a flow."""
    chunks = []
    chunks.append(_build_flow_overview(flow))
    chunks.extend(_build_menu_option_chunks(flow))
    chunks.extend(_build_data_table_chunks(flow))
    chunks.extend(_build_dynamic_group_chunks(flow))
    chunks.extend(_build_task_flow_chunks(flow))
    return chunks


def _get_stored_flow_version(collection, flow_id: str) -> dict:
    """Read published_version_id/date/publisher from the flow_overview chunk in ChromaDB."""
    chunk_id = _safe_id(flow_id, "overview")
    try:
        results = collection.get(ids=[chunk_id], include=["metadatas"])
        if results and results.get("metadatas") and results["metadatas"]:
            meta = results["metadatas"][0]
            return {
                "version_id":     meta.get("published_version_id", ""),
                "published_date": meta.get("published_date", ""),
                "published_by_id": meta.get("published_by_id", ""),
            }
    except Exception as e:
        logger.debug("_get_stored_flow_version failed for %s: %s", flow_id, e)
    return {}


def _diff_chunks(old_chunks: list, new_chunks: list) -> dict:
    """Compute added / removed / modified between two chunk lists."""
    old_map = {c["chunk_id"]: c for c in old_chunks}
    new_map = {c["chunk_id"]: c for c in new_chunks}

    added   = [c for cid, c in new_map.items() if cid not in old_map]
    removed = [c for cid, c in old_map.items() if cid not in new_map]
    modified = [
        {
            "chunk_id":   cid,
            "chunk_type": new_map[cid].get("chunk_type", ""),
            "label":      new_map[cid].get("task_name", new_map[cid].get("entity_name", cid)),
            "old_text":   old_map[cid]["text"],
            "new_text":   new_map[cid]["text"],
        }
        for cid in new_map
        if cid in old_map and old_map[cid]["text"] != new_map[cid]["text"]
    ]
    return {"added": added, "removed": removed, "modified": modified}


_user_name_cache: dict = {}

def _resolve_user_name(user_id: str) -> str:
    """Resolve a Genesys user ID to a display name. Cached per process run."""
    if not user_id:
        return "Unknown"
    if user_id in _user_name_cache:
        return _user_name_cache[user_id]
    try:
        data = _get(f"/users/{user_id}")
        name = data.get("name", "") or data.get("displayName", "")
        if not name:
            name = f"{data.get('givenName','')} {data.get('familyName','')}".strip()
        _user_name_cache[user_id] = name or user_id
    except Exception as e:
        logger.debug("Could not resolve user %s: %s", user_id, e)
        _user_name_cache[user_id] = user_id
    return _user_name_cache[user_id]


def run_flow_change_detection(flow_id: str, collection) -> dict:
    """
    Compare the currently published Genesys version against what is stored in ChromaDB.

    Returns {changed: bool}. If changed=True, also includes:
      - flow_name, flow_id, old_version, new_version, published_date
      - published_by (resolved display name)
      - diff: {added, removed, modified}
      - new_chunks: ready to upsert into ChromaDB
    """
    meta = _get(f"/flows/{flow_id}")
    if "_error" in meta:
        return {"changed": False, "error": meta["_error"], "flow_id": flow_id}

    pub  = meta.get("publishedVersion") or {}
    current_version_id   = pub.get("id", "")
    current_publish_date = pub.get("datePublished", "")
    current_publisher_id = (pub.get("createdBy") or {}).get("id", "")
    flow_name            = meta.get("name", flow_id)

    if not current_version_id:
        return {"changed": False, "flow_name": flow_name, "reason": "no published version"}

    stored = _get_stored_flow_version(collection, flow_id)

    if stored.get("version_id") and stored["version_id"] == current_version_id:
        return {"changed": False, "flow_name": flow_name, "version_id": current_version_id}

    logger.info("Change detected: %s  %s -> %s",
                flow_name, stored.get("version_id", "none"), current_version_id)

    flow_base = {
        "flow_id":              flow_id,
        "flow_name":            flow_name,
        "flow_type":            meta.get("type", ""),
        "division":             meta.get("division", {}).get("name", "") if isinstance(meta.get("division"), dict) else "",
        "description":          meta.get("description", ""),
        "published_version_id": current_version_id,
        "published_date":       current_publish_date,
        "published_by_id":      current_publisher_id,
    }

    try:
        new_cfg    = _get(f"/flows/{flow_id}/latestconfiguration")
        new_flow   = {**flow_base, "config": new_cfg}
        new_chunks = _build_flow_chunks(new_flow)
    except Exception as e:
        return {
            "changed":        True,
            "flow_name":      flow_name,
            "flow_id":        flow_id,
            "new_version":    current_version_id,
            "published_date": current_publish_date,
            "error":          f"failed to fetch new config: {e}",
        }

    old_chunks = []
    if stored.get("version_id"):
        try:
            old_cfg  = _get(f"/flows/{flow_id}/versions/{stored['version_id']}/configuration")
            old_flow = {
                **flow_base,
                "published_version_id": stored["version_id"],
                "published_date":       stored.get("published_date", ""),
                "config":               old_cfg,
            }
            old_chunks = _build_flow_chunks(old_flow)
        except Exception as e:
            logger.warning("Could not fetch old config for %s (%s): %s",
                           flow_name, stored["version_id"], e)

    diff         = _diff_chunks(old_chunks, new_chunks)
    published_by = _resolve_user_name(current_publisher_id)

    return {
        "changed":         True,
        "flow_name":       flow_name,
        "flow_id":         flow_id,
        "old_version":     stored.get("version_id", "unknown"),
        "new_version":     current_version_id,
        "published_date":  current_publish_date,
        "published_by":    published_by,
        "published_by_id": current_publisher_id,
        "diff":            diff,
        "new_chunks":      new_chunks,
    }


def run_org_entities_refresh(status_dict: Optional[dict] = None) -> dict:
    """
    Re-ingest all org-level entities (queues, groups, data tables, prompts, IVRs,
    DID pools, schedule groups, schedules, wrap-up codes, recording policies).
    Does NOT touch flow config chunks.
    """
    def update(msg):
        logger.info(msg)
        if status_dict is not None:
            status_dict["message"] = msg

    col = _get_collection()
    total_by_type: dict = {}

    def flush(chunks, label):
        if not chunks:
            return
        _upsert(col, chunks)
        for c in chunks:
            ct = c["chunk_type"]
            total_by_type[ct] = total_by_type.get(ct, 0) + 1
        update(f"  {label}: {len(chunks)} chunks upserted")

    update("Refreshing org entities...")

    update("Fetching queues...")
    queues = _fetch_all("/routing/queues", {"expand": "memberCount,queueFlow,callingPartyName,callingPartyNumber"})

    update("Fetching wrap-up codes...")
    wrapup = _fetch_all("/routing/wrapupcodes")

    update(f"Fetching wrapup-to-queue assignments for {len(queues)} queues...")
    queue_ids    = [q["id"] for q in queues if q.get("id")]
    q_wrapup_raw = _fetch_queue_wrapup_map(queue_ids)
    q_id_to_name = {q["id"]: q["name"] for q in queues if q.get("id") and q.get("name")}
    wrapup_to_queue_names: dict = {}
    for qid, codes in q_wrapup_raw.items():
        for c in codes:
            cid = c.get("id", "")
            if cid:
                wrapup_to_queue_names.setdefault(cid, [])
                q_name = q_id_to_name.get(qid, qid)
                if q_name not in wrapup_to_queue_names[cid]:
                    wrapup_to_queue_names[cid].append(q_name)

    flush(_chunks_queues(queues, queue_wrapup_map=q_wrapup_raw), f"queues ({len(queues)})")
    flush(_chunks_wrapup_codes(wrapup, wrapup_to_queues=wrapup_to_queue_names), f"wrap-up codes ({len(wrapup)})")

    update("Fetching groups...")
    groups = _fetch_all("/groups", {"sortOrder": "ASC"})
    flush(_chunks_groups(groups), f"groups ({len(groups)})")

    update("Fetching data tables...")
    tables = _fetch_all("/flows/datatables")
    flush(_chunks_datatables(tables), f"data tables ({len(tables)})")

    update("Fetching prompts...")
    prompts = _fetch_all("/architect/prompts")
    flush(_chunks_prompts(prompts), f"prompts ({len(prompts)})")

    update("Fetching IVRs...")
    ivrs = _fetch_all("/architect/ivrs")
    flush(_chunks_ivrs(ivrs), f"IVRs ({len(ivrs)})")

    update("Fetching DID pools...")
    did_pools = _fetch_all("/telephony/providers/edges/didpools")
    flush(_chunks_did_pools(did_pools), f"DID pools ({len(did_pools)})")

    update("Fetching schedule groups...")
    sched_groups = _fetch_all("/architect/schedulegroups")
    flush(_chunks_schedule_groups(sched_groups), f"schedule groups ({len(sched_groups)})")

    update("Fetching schedules...")
    schedules = _fetch_all("/architect/schedules")
    flush(_chunks_schedules(schedules), f"schedules ({len(schedules)})")

    update("Fetching recording policies...")
    rec_policies = _fetch_all("/recording/mediaretentionpolicies")
    w_name_map   = {w["id"]: w["name"] for w in wrapup if w.get("id") and w.get("name")}
    flush(_chunks_recording_policies(rec_policies, q_id_to_name, w_name_map), f"recording policies ({len(rec_policies)})")

    update(f"Org entity refresh complete: {sum(total_by_type.values())} chunks")
    return {"by_type": total_by_type, "queue_count": len(queues), "group_count": len(groups)}


# =========================================================================
# INGEST ENTRY POINTS
# =========================================================================

def run_ingest(flow_name: str, reset: bool = False, status_dict: Optional[dict] = None) -> dict:
    """Ingest a single flow by name or UUID."""
    def update(msg):
        logger.info(msg)
        if status_dict is not None:
            status_dict["message"] = msg

    update(f"Fetching flow configuration: {flow_name}")
    flow   = _fetch_flow_config(flow_name)
    update(f"Fetched '{flow['flow_name']}' ({flow['flow_type']}), parsing chunks...")
    chunks = _build_flow_chunks(flow)

    update(f"Built {len(chunks)} chunks, upserting to ChromaDB...")
    col = _get_collection()
    if reset:
        col.delete(where={"flow_id": flow["flow_id"]})
    _upsert(col, chunks)

    by_type: dict = {}
    for c in chunks:
        ct = c["chunk_type"]
        by_type[ct] = by_type.get(ct, 0) + 1

    summary = {
        "flow_name": flow["flow_name"],
        "flow_id": flow["flow_id"],
        "total_chunks": len(chunks),
        "by_type": by_type,
    }
    update(f"Ingest complete: {summary}")
    return summary


def run_org_ingest(reset: bool = False, status_dict: Optional[dict] = None) -> dict:
    """Full org-wide ingest: all flows, queues, groups, data tables, and all other entities."""
    def update(msg):
        logger.info(msg)
        if status_dict is not None:
            status_dict["message"] = msg

    col = _get_collection()
    if reset:
        update("Resetting entire collection...")
        col.delete(where={})

    total_by_type: dict = {}
    full_config_ids: set = set()

    def flush(chunks, label):
        if not chunks:
            return
        _upsert(col, chunks)
        for c in chunks:
            ct = c["chunk_type"]
            total_by_type[ct] = total_by_type.get(ct, 0) + 1
        update(f"  {label}: {len(chunks)} chunks upserted")

    update("Fetching all flows...")
    all_flows = _fetch_all("/flows", {"publishedOnly": "true"})
    update(f"Found {len(all_flows)} published flows")

    FULL_CONFIG_TYPES = {
        "inboundcall", "inboundshortmessage", "inboundchat",
        "commonmodule",
        "inqueuecall", "inqueueshortmessage",
        "bot", "digitalbot",
        "outboundcall",
        "voicemail",
        "workflow",
    }
    inbound_flows = [
        f for f in all_flows
        if f.get("type", "").lower() in FULL_CONFIG_TYPES
        and f.get("active", True)
        and "healthcheck" not in f.get("name", "").lower()
        and "test"        not in f.get("name", "").lower()
        and "_bu"         not in f.get("name", "").lower()
        and "backup"      not in f.get("name", "").lower()
    ]
    flush(_chunks_flow_metadata(all_flows, set(f["id"] for f in inbound_flows)), "flow metadata (non-IVR)")

    update(f"Fetching full config for {len(inbound_flows)} flows...")
    for i, f in enumerate(inbound_flows):
        fid = f["id"]
        full_config_ids.add(fid)
        update(f"  Flow {i+1}/{len(inbound_flows)}: {f['name']}")
        try:
            flow   = _fetch_flow_config(fid)
            chunks = _build_flow_chunks(flow)
            flush(chunks, f"flow '{f['name']}'")
        except Exception as e:
            update(f"  WARNING: could not ingest flow {f['name']}: {e}")
        time.sleep(0.2)

    update("Fetching queues...")
    queues = _fetch_all("/routing/queues", {"expand": "memberCount,queueFlow,callingPartyName,callingPartyNumber"})
    update("Fetching wrap-up codes...")
    wrapup = _fetch_all("/routing/wrapupcodes")
    update(f"Fetching wrapup-to-queue assignments for {len(queues)} queues...")
    q_id_to_name = {q["id"]: q["name"] for q in queues if q.get("id") and q.get("name")}
    queue_ids = [q["id"] for q in queues if q.get("id")]
    q_wrapup_raw = _fetch_queue_wrapup_map(queue_ids)
    wrapup_to_queue_names: dict = {}
    for qid, codes in q_wrapup_raw.items():
        for c in codes:
            cid = c.get("id", "")
            if cid:
                wrapup_to_queue_names.setdefault(cid, [])
                q_name = q_id_to_name.get(qid, qid)
                if q_name not in wrapup_to_queue_names[cid]:
                    wrapup_to_queue_names[cid].append(q_name)
    flush(_chunks_queues(queues, queue_wrapup_map=q_wrapup_raw), f"queues ({len(queues)})")
    flush(_chunks_wrapup_codes(wrapup, wrapup_to_queues=wrapup_to_queue_names), f"wrap-up codes ({len(wrapup)})")

    update("Fetching groups...")
    groups = _fetch_all("/groups", {"sortOrder": "ASC"})
    flush(_chunks_groups(groups), f"groups ({len(groups)})")

    update("Fetching data tables...")
    tables = _fetch_all("/flows/datatables")
    flush(_chunks_datatables(tables), f"data tables ({len(tables)})")

    update("Fetching prompts...")
    prompts = _fetch_all("/architect/prompts")
    flush(_chunks_prompts(prompts), f"prompts ({len(prompts)})")

    update("Fetching IVRs...")
    ivrs = _fetch_all("/architect/ivrs")
    flush(_chunks_ivrs(ivrs), f"IVRs ({len(ivrs)})")

    update("Fetching DID pools...")
    did_pools = _fetch_all("/telephony/providers/edges/didpools")
    flush(_chunks_did_pools(did_pools), f"DID pools ({len(did_pools)})")

    update("Fetching schedule groups...")
    sched_groups = _fetch_all("/architect/schedulegroups")
    flush(_chunks_schedule_groups(sched_groups), f"schedule groups ({len(sched_groups)})")

    update("Fetching schedules...")
    schedules = _fetch_all("/architect/schedules")
    flush(_chunks_schedules(schedules), f"schedules ({len(schedules)})")

    update("Fetching recording policies...")
    rec_policies = _fetch_all("/recording/mediaretentionpolicies")
    w_name_map = {w["id"]: w["name"] for w in wrapup if w.get("id") and w.get("name")}
    flush(_chunks_recording_policies(rec_policies, q_id_to_name, w_name_map), f"recording policies ({len(rec_policies)})")

    total = sum(total_by_type.values())
    summary = {
        "scope": "full_org",
        "total_chunks": total,
        "by_type": total_by_type,
        "flow_count": len(all_flows),
        "inbound_flow_config_count": len(inbound_flows),
        "queue_count": len(queues),
        "group_count": len(groups),
        "datatable_count": len(tables),
        "ivr_count": len(ivrs),
    }
    update(f"Org ingest complete: {total} total chunks")
    return summary
