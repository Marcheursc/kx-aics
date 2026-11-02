#!/bin/bash
# 快修到家 AI 客服 — 一键启动/停止脚本

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_DIR/frontend"
PID_DIR="$PROJECT_DIR/.pids"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
STORAGE_DIR="${CHROMA_STORAGE:-$HOME/.kx-aics/storage}"
INDEX_DB="$STORAGE_DIR/chroma.sqlite3"

mkdir -p "$PID_DIR"

# 索引是否需要重建：不存在，或 data/ 下有比索引更新的语料
index_stale() {
    [ -f "$INDEX_DB" ] || return 0
    local newer
    newer=$(find "$PROJECT_DIR/data" -maxdepth 1 -type f \
        \( -name '*.txt' -o -name '*.md' \) -newer "$INDEX_DB" -print -quit 2>/dev/null)
    [ -n "$newer" ] && return 0
    return 1
}

ensure_index() {
    if index_stale; then
        if [ -f "$INDEX_DB" ]; then
            echo "[0/2] 检测到 data/ 语料有更新，重建知识库索引 ..."
        else
            echo "[0/2] 未找到知识库索引，首次构建（约 30 秒）..."
        fi
        cd "$PROJECT_DIR"
        "$PYTHON" -m src.ingest || { echo "[错误] 索引构建失败，请检查上方日志"; exit 1; }
        echo ""
    else
        echo "[0/2] 知识库索引已是最新，跳过构建"
    fi
}

start() {
    echo "========================================"
    echo "  快修到家 AI 客服 — 启动中..."
    echo "========================================"

    # 停止旧进程
    stop 2>/dev/null

    ensure_index

    # 启动后端
    echo "[1/2] 启动后端服务 (FastAPI :8000)..."
    cd "$PROJECT_DIR"
    setsid nohup "$PYTHON" -m uvicorn src.server:app --host 0.0.0.0 --port 8000 \
        > /tmp/ai-cs-backend.log 2>&1 < /dev/null &
    echo $! > "$PID_DIR/backend.pid"
    echo "      后端 PID: $(cat $PID_DIR/backend.pid)"

    # 启动前端
    echo "[2/2] 启动前端服务 (Vite :5173)..."
    cd "$FRONTEND_DIR"
    setsid nohup "$FRONTEND_DIR/node_modules/.bin/vite" --host \
        > /tmp/ai-cs-frontend.log 2>&1 < /dev/null &
    echo $! > "$PID_DIR/frontend.pid"
    echo "      前端 PID: $(cat $PID_DIR/frontend.pid)"

    # 后端要加载 BGE/Reranker 模型，最多等 120 秒
    echo ""
    echo "      等待后端模型加载 ..."
    for _ in $(seq 1 60); do
        if curl -s --max-time 2 http://localhost:8000/api/health > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    if curl -s --max-time 3 http://localhost:8000/api/health > /dev/null 2>&1 && \
       kill -0 "$(cat $PID_DIR/frontend.pid)" 2>/dev/null; then
        echo ""
        echo "========================================"
        echo "  启动成功！"
        echo "  前端: http://localhost:5173"
        echo "  后端: http://localhost:8000"
        echo "  API文档: http://localhost:8000/docs"
        echo "========================================"
    else
        echo "[错误] 服务启动异常，请检查日志:"
        echo "  后端日志: /tmp/ai-cs-backend.log"
        echo "  前端日志: /tmp/ai-cs-frontend.log"
        echo "  最近 15 行后端日志:"
        tail -15 /tmp/ai-cs-backend.log
        exit 1
    fi
}

stop() {
    echo "========================================"
    echo "  快修到家 AI 客服 — 关闭中..."
    echo "========================================"

    for name in backend frontend; do
        pid_file="$PID_DIR/${name}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
                echo "  已关闭 $name (PID: $pid)"
            fi
            rm -f "$pid_file"
        fi
    done

    # 清理残留
    fuser -k 8000/tcp 2>/dev/null && echo "  已清理残留 8000 端口进程" || true
    fuser -k 5173/tcp 2>/dev/null && echo "  已清理残留 5173 端口进程" || true

    echo "  全部关闭完成"
}

status() {
    echo "========================================"
    echo "  快修到家 AI 客服 — 服务状态"
    echo "========================================"

    backend_ok=0
    frontend_ok=0

    if [ -f "$PID_DIR/backend.pid" ] && kill -0 "$(cat $PID_DIR/backend.pid)" 2>/dev/null; then
        echo "  后端 (FastAPI) :8000   ✓ 运行中 (PID: $(cat $PID_DIR/backend.pid))"
        backend_ok=1
    else
        echo "  后端 (FastAPI) :8000   ✗ 未运行"
    fi

    if [ -f "$PID_DIR/frontend.pid" ] && kill -0 "$(cat $PID_DIR/frontend.pid)" 2>/dev/null; then
        echo "  前端 (Vite)   :5173   ✓ 运行中 (PID: $(cat $PID_DIR/frontend.pid))"
        frontend_ok=1
    else
        echo "  前端 (Vite)   :5173   ✗ 未运行"
    fi

    echo ""
    if [ -f "$INDEX_DB" ]; then
        echo "  知识库索引: ✓ $(du -h "$INDEX_DB" | cut -f1)  更新于 $(date -r "$INDEX_DB" '+%Y-%m-%d %H:%M')"
        if index_stale; then
            echo "               ⚠ data/ 语料比索引新，运行 ./run.sh reindex 重建"
        fi
    else
        echo "  知识库索引: ✗ 不存在，运行 ./run.sh reindex 构建"
    fi

    if [ $backend_ok -eq 1 ] && [ $frontend_ok -eq 1 ]; then
        echo ""
        echo "  访问地址: http://localhost:5173"
    fi
}

reindex() {
    echo "========================================"
    echo "  快修到家 AI 客服 — 重建知识库索引"
    echo "========================================"
    cd "$PROJECT_DIR"
    "$PYTHON" -m src.ingest
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    status)  status ;;
    reindex) reindex ;;
    restart)
        stop
        sleep 1
        start
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|reindex}"
        echo ""
        echo "  start    启动前后端（索引缺失或过期时自动重建）"
        echo "  stop     关闭前后端"
        echo "  restart  重启"
        echo "  status   查看服务与索引状态"
        echo "  reindex  强制重建知识库索引"
        exit 1
        ;;
esac
