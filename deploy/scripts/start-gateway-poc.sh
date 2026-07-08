#!/usr/bin/env bash
# start-gateway.sh
# 为 310P 128K vllm 服务启动 llm-service gateway 层（端口 8001）。
#
# 前提：
#   - vllm-qwen36-128k 容器已在 localhost:18082 运行
#   - 已在本仓库安装依赖（uv sync 或 pip install -e .）
#
# 用法：
#   ./start-gateway.sh          # 前台运行（Ctrl+C 停止）
#   ./start-gateway.sh --check  # 仅检查依赖，不启动

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 优先使用 venv 里的 Python
PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [ ! -x "${PYTHON}" ]; then
    PYTHON="python3"
fi

# ── 检查依赖 ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ "${1:-}" = "--check" ]; then
    log "检查依赖..."
    "${PYTHON}" -c "import llm_service; print('  llm_service OK')"
    "${PYTHON}" -c "import fastapi; print('  fastapi OK')"
    "${PYTHON}" -c "import uvicorn; print('  uvicorn OK')"
    log "✅ 依赖检查通过"
    exit 0
fi

# ── 检查 vllm 是否就绪 ────────────────────────────────────────────────────────
log "检查 vllm 服务（localhost:18082）..."
if ! curl -sf http://localhost:18082/v1/models | grep -q '"id"'; then
    log "❌ vllm 服务未就绪，请先确认 vllm-qwen36-128k 容器正常运行"
    exit 1
fi
MODEL=$(curl -s http://localhost:18082/v1/models | "${PYTHON}" -c \
    "import sys,json; r=json.load(sys.stdin); print(r['data'][0]['id'])" 2>/dev/null)
log "✅ vllm 就绪，模型: ${MODEL}"

# ── 检查 gateway 依赖 ─────────────────────────────────────────────────────────
if ! "${PYTHON}" -c "import llm_service" 2>/dev/null; then
    log "llm_service 未安装，尝试安装依赖..."
    if command -v uv >/dev/null 2>&1; then
        uv sync
        PYTHON="${REPO_ROOT}/.venv/bin/python3"
    else
        pip install -e . -q
    fi
fi

# ── 启动 gateway ──────────────────────────────────────────────────────────────
log "启动 gateway（端口 8001）..."
log "  thinking 默认关闭（可通过 enable_thinking: true 按需开启）"
log "  转发至 http://127.0.0.1:18082/v1"
log "  客户端接入: http://localhost:8001/v1"
log ""

exec env \
    PORT=8001 \
    BACKENDS_CONFIG_PATH=configs/backends.310p-128k.yaml \
    "${PYTHON}" main.py
