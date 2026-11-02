import os
import uuid
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys_path = str(PROJECT_ROOT / "src")

import sys

sys.path.insert(0, sys_path)
from chain import CustomerServiceChain, get_pool
from graph import get_intent_graph

sessions: dict[str, dict] = {}
graph_sessions: dict[str, dict] = {}
SESSION_TTL = 3600


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict] = []


def _get_or_create_chain(session_id: Optional[str]) -> tuple[CustomerServiceChain, str]:
    now = time.time()
    if session_id and session_id in sessions:
        sessions[session_id]["last_access"] = now
        return sessions[session_id]["chain"], session_id

    new_id = str(uuid.uuid4())[:8]
    chain = CustomerServiceChain()
    sessions[new_id] = {"chain": chain, "last_access": now}
    return chain, new_id


def _cleanup_expired():
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["last_access"] > SESSION_TTL]
    for sid in expired:
        del sessions[sid]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] 正在预热模型池 ...")
    get_pool()
    get_intent_graph()
    print("[Server] 快修到家 AI 客服 API 服务启动\n")
    yield
    print("[Server] 服务关闭")


app = FastAPI(title="AI智能客服系统", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    _cleanup_expired()
    chain, sid = _get_or_create_chain(req.session_id)

    try:
        answer = chain.answer(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回答生成失败: {str(e)}")

    return ChatResponse(answer=answer, session_id=sid, sources=chain.last_sources)


@app.post("/api/chat/graph", response_model=ChatResponse)
def chat_graph(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    _cleanup_expired()
    now = time.time()

    if req.session_id and req.session_id in graph_sessions:
        graph_sessions[req.session_id]["last_access"] = now
        graph_obj = graph_sessions[req.session_id]["graph"]
    else:
        sid = str(uuid.uuid4())[:8]
        graph_obj = get_intent_graph()
        graph_sessions[sid] = {"graph": graph_obj, "last_access": now}
        req.session_id = sid

    try:
        answer = graph_obj.answer(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回答生成失败: {str(e)}")

    return ChatResponse(answer=answer, session_id=req.session_id, sources=graph_obj.last_sources)


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        sessions[session_id]["chain"].clear_memory()
        return {"status": "cleared"}
    raise HTTPException(status_code=404, detail="会话不存在")


FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
