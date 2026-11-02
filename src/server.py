import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chain import CustomerServiceChain, get_pool
from graph import IntentGraph

SESSION_TTL = 3600

# sid -> {"kind": "chain"|"graph", "obj": 会话对象, "last_access": 时间戳}
SESSIONS: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict] = []


def _cleanup_expired() -> None:
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s["last_access"] > SESSION_TTL]
    for sid in expired:
        SESSIONS.pop(sid, None)


def _get_or_create(session_id: Optional[str], kind: str, factory) -> tuple[object, str]:
    """按 sid + 会话类型取回对象；类型不符或不存在则新建独立实例。"""
    if session_id and session_id in SESSIONS and SESSIONS[session_id]["kind"] == kind:
        SESSIONS[session_id]["last_access"] = time.time()
        return SESSIONS[session_id]["obj"], session_id

    sid = str(uuid.uuid4())[:8]
    obj = factory()
    SESSIONS[sid] = {"kind": kind, "obj": obj, "last_access": time.time()}
    return obj, sid


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] 正在预热模型池 ...")
    get_pool()
    print("[Server] 快修到家 AI 客服 API 服务启动\n")
    yield
    print("[Server] 服务关闭")


app = FastAPI(title="快修到家 AI 客服系统", version="2.1.0", lifespan=lifespan)

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # 通配来源不能与凭据共用，否则浏览器会拒绝请求；本项目用 session_id 而非 Cookie
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    _cleanup_expired()
    chain, sid = _get_or_create(req.session_id, "chain", CustomerServiceChain)

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
    graph_obj, sid = _get_or_create(req.session_id, "graph", IntentGraph)

    try:
        answer = graph_obj.answer(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回答生成失败: {str(e)}")

    return ChatResponse(answer=answer, session_id=sid, sources=graph_obj.last_sources)


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    entry = SESSIONS.pop(session_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    obj = entry["obj"]
    if hasattr(obj, "clear_memory"):
        obj.clear_memory()
    return {"status": "cleared", "kind": entry["kind"]}


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
