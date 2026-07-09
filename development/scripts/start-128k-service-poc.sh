#!/usr/bin/env bash
# start-128k-service.sh
#
# 启动 Qwen3.6-35B-A3B-w8a8 128K 长上下文推理服务（310P 双卡）
#
# 用法:
#   ./start-128k-service.sh              # 后台启动，日志实时打印到终端
#   ./start-128k-service.sh --detach     # 后台启动，日志不打印
#   ./start-128k-service.sh --stop       # 停止服务容器
#   ./start-128k-service.sh --status     # 查看容器状态
#   ./start-128k-service.sh --logs       # 查看服务日志（tail -f）
#
# 说明:
#   - max_num_batched_tokens=1024 是精度关键参数，不可改回 2048
#     （2048 会导致 14k+ tokens 的 ChunkedPrefill 输出乱码，根因见 docs/development/WORK_SUMMARY_20260626.md）
#   - 服务就绪后监听 localhost:18082
#   - 模型别名: qwen3.6-128k
#   - 启动耗时约 6-8 分钟（冷启动含 torch.compile）

set -e

# ── 基础配置 ──────────────────────────────────────────────────────────────────
CONTAINER_NAME="vllm-qwen36-128k"
IMAGE="registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:26.0.0-poc-300i-duo-py311-ubuntu24.04-arm64"
PORT=18082
SERVED_MODEL_NAME="qwen3.6-128k"

# 路径配置（相对脚本所在目录自动解析）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}"
MODELS_DIR="/srv/meetai/models"

# ── 命令行解析 ────────────────────────────────────────────────────────────────
ACTION="start"
for arg in "$@"; do
    case "$arg" in
        --detach|-d)  ACTION="start-detach" ;;
        --stop)       ACTION="stop" ;;
        --status)     ACTION="status" ;;
        --logs)       ACTION="logs" ;;
        --help|-h)    ACTION="help" ;;
    esac
done

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_ready() {
    local start elapsed
    start=$(date +%s)
    log "等待服务就绪（端口 ${PORT}）..."
    while true; do
        elapsed=$(( $(date +%s) - start ))
        if curl -sf "http://localhost:${PORT}/v1/models" | grep -q '"id"'; then
            log "✅ 服务就绪！耗时 ${elapsed}s"
            curl -s "http://localhost:${PORT}/v1/models" | python3 -c "
import sys, json
r = json.load(sys.stdin)
for m in r.get('data', []):
    print(f\"  模型: {m['id']}  max_model_len: {m.get('max_model_len', 'N/A')}\")
"
            return 0
        fi
        if [ $elapsed -gt 600 ]; then
            log "❌ 超时（600s），请检查日志: sudo podman logs ${CONTAINER_NAME}"
            return 1
        fi
        # 每 15s 打印一次最新日志
        last_log=$(sudo podman logs --tail 1 "${CONTAINER_NAME}" 2>&1 | tail -1)
        echo "  [${elapsed}s] ${last_log}"
        sleep 15
    done
}

# ── help ──────────────────────────────────────────────────────────────────────
if [ "$ACTION" = "help" ]; then
    sed -n '2,15p' "$0" | sed 's/^# \?//'
    exit 0
fi

# ── status ────────────────────────────────────────────────────────────────────
if [ "$ACTION" = "status" ]; then
    sudo podman ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    curl -sf "http://localhost:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null \
        || echo "（服务未响应）"
    exit 0
fi

# ── logs ──────────────────────────────────────────────────────────────────────
if [ "$ACTION" = "logs" ]; then
    exec sudo podman logs -f "${CONTAINER_NAME}"
fi

# ── stop ──────────────────────────────────────────────────────────────────────
if [ "$ACTION" = "stop" ]; then
    log "停止容器 ${CONTAINER_NAME}..."
    sudo podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    sudo podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
    log "✅ 已停止"
    exit 0
fi

# ── start ─────────────────────────────────────────────────────────────────────

# 检查 patches 文件存在
for f in metadata_builder.py attention_v1.py attention_mask.py; do
    if [ ! -f "${PATCHES_DIR}/${f}" ]; then
        log "❌ 缺少 patch 文件: ${PATCHES_DIR}/${f}"
        exit 1
    fi
done

# 清理旧容器（如果存在）
if sudo podman ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "发现已有容器 ${CONTAINER_NAME}，停止并清理..."
    sudo podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    sudo podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
fi

log "启动容器 ${CONTAINER_NAME}..."
log "  Image:   ${IMAGE}"
log "  Port:    ${PORT}"
log "  Model:   ${SERVED_MODEL_NAME} (max_model_len=131072)"
log "  Patches: ${PATCHES_DIR}"

sudo podman run -d \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --network host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v "${MODELS_DIR}:/models:ro" \
    -v "${PATCHES_DIR}:/workspace/patches:ro" \
    "${IMAGE}" \
    bash -c '
        set -e

        # 1. 设置 CANN 环境
        source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

        # 2. 定位 vllm_ascend 路径并应用 patches
        VLLM_PATH=$(python3 -c "import vllm_ascend; import os; print(os.path.dirname(vllm_ascend.__file__))")
        echo "[INFO] vllm_ascend path: $VLLM_PATH"

        cp /workspace/patches/metadata_builder.py $VLLM_PATH/_310p/attention/
        cp /workspace/patches/attention_v1.py      $VLLM_PATH/_310p/attention/
        cp /workspace/patches/attention_mask.py    $VLLM_PATH/_310p/attention/
        echo "[INFO] Patches applied successfully"

        # 3. 启动 vllm 服务（128K）
        #
        # 关键参数说明:
        #   --max-num-batched-tokens 1024  精度修复参数（2048 会导致 14k+ 乱码）
        #   --max-num-seqs 1              单并发，保守稳定配置
        #   --reasoning-parser qwen3      thinking 内容分离到 reasoning 字段
        #   --enable-chunked-prefill      必须开启，配合 max-num-batched-tokens
        #   --no-enable-prefix-caching    310P 上 prefix caching 未验证
        #   --compilation-config FULL_DECODE_ONLY  decode 图模式，+12% decode 速度
        #   --mamba-ssm-cache-dtype float16        线性注意力状态 dtype
        exec vllm serve /models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8 \
            --served-model-name '"${SERVED_MODEL_NAME}"' \
            --host 0.0.0.0 \
            --port '"${PORT}"' \
            -tp 2 \
            --max-model-len 131072 \
            --max-num-seqs 1 \
            --max-num-batched-tokens 1024 \
            --gpu-memory-utilization 0.75 \
            --dtype float16 \
            --kv-cache-dtype auto \
            --trust-remote-code \
            --enable-chunked-prefill \
            --no-enable-prefix-caching \
            --reasoning-parser qwen3 \
            --additional-config "{\"ascend_compilation_config\": {\"fuse_norm_quant\": false}}" \
            --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [1]}" \
            --async-scheduling \
            --mamba-ssm-cache-dtype float16 \
            --allowed-local-media-path /
    '

log "容器已启动，等待服务就绪..."

if [ "$ACTION" = "start" ]; then
    wait_ready
elif [ "$ACTION" = "start-detach" ]; then
    log "后台模式，跳过等待。查看日志: $0 --logs"
fi
