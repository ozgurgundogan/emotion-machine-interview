import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.client import ToolSelectorClient
from src.environment import INDEX_PATH, METADATA_PATH, RESPONSE_RETRIEVAL_COUNT

app = FastAPI(title="Tool Selector Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


AGENT = ToolSelectorClient(index_path=INDEX_PATH, metadata_path=METADATA_PATH)

async def stream_output(plan):
    text = json.dumps(plan, indent=2, ensure_ascii=False)
    for line in text.splitlines():
        yield (line + "\n").encode("utf-8")
        await asyncio.sleep(0.01)


@app.post("/api/query")
async def query_tool(req: Request):
    context_session_id = str(uuid.uuid4())
    data = await req.json()
    query = data.get("query", "")
    stream = bool(data.get("stream"))
    result = AGENT.plan_query(query, count=RESPONSE_RETRIEVAL_COUNT)

    if stream:
        return StreamingResponse(stream_output(result), media_type="text/plain")
    return JSONResponse(result)
