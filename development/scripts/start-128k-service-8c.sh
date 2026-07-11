#!/usr/bin/env bash
# start-128k-service-8c.sh
#
# 启动 Qwen3.6-35B-A3B-w8a8 128K 长上下文推理服务（310P 双卡，8 并发优化版）
# 镜像：310p-opt-20260708（内置 310P patches，无需外部挂载）
#
# 用法:
#   ./start-128k-service-8c.sh              # 启动并等待就绪
#   ./start-128k-service-8c.sh --detach     # 后台启动，不等待
#   ./start-128k-service-8c.sh --stop       # 停止服务容器
#   ./start-128k-service-8c.sh --status     # 查看容器状态
#   ./start-128k-service-8c.sh --logs       # 查看服务日志（tail -f）
#
# 关键参数说明:
#   --max-num-batched-tokens 2048
#       精度已验证正常（旧 1024 约束仅适用于旧 poc-300i-duo 镜像）
#
#   --max-num-seqs 8
#       支持最大 8 个并发请求
#
#   cudagraph_capture_sizes=[1,2,4,8]
#       为并发 Decode 预编译图，避免 concurrency≥2 时退回 eager 模式
#       （[1] 时 concurrency≥2 延迟从 ~32ms/token 膨胀至 ~245ms/token）
#
#   kv-cache-memory 21653924864
#       固定 KV cache 显存（~20GB），替代 gpu-memory-utilization 比例分配
#
#   --reasoning-parser qwen3
#       thinking 内容分离到 reasoning 字段

set -e

# ── 基础配置 ──────────────────────────────────────────────────────────────────
CONTAINER_NAME="vllm-310p-opt-128k-8c"
IMAGE="registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708"
PORT=18082
SERVED_MODEL_NAME="qwen3.6-128k-8c"
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
            log "服务就绪！耗时 ${elapsed}s"
            curl -s "http://localhost:${PORT}/v1/models" | python3 -c "
import sys, json
r = json.load(sys.stdin)
for m in r.get('data', []):
    print(f\"  模型: {m['id']}  max_model_len: {m.get('max_model_len', 'N/A')}\")
"
            return 0
        fi
        if [ $elapsed -gt 600 ]; then
            log "超时（600s），请检查日志: sudo podman logs ${CONTAINER_NAME}"
            return 1
        fi
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
    log "已停止"
    exit 0
fi

# ── start ─────────────────────────────────────────────────────────────────────

# 清理旧容器（如果存在）
if sudo podman ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "发现已有容器 ${CONTAINER_NAME}，停止并清理..."
    sudo podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    sudo podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
fi

log "启动容器 ${CONTAINER_NAME}..."
log "  Image:  ${IMAGE}"
log "  Port:   ${PORT}"
log "  Model:  ${SERVED_MODEL_NAME} (max_model_len=131072)"
log "  cudagraph_capture_sizes: [1, 2, 4, 8]"

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
    "${IMAGE}" \
    bash -c '
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        export ASCEND_CUSTOM_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom

        python3 -m vllm.entrypoints.openai.api_server \
            --model /models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8 \
            --served-model-name qwen3.6-128k-8c \
            --host 0.0.0.0 \
            --port 18082 \
            -tp 2 \
            --max-model-len 131072 \
            --max-num-seqs 8 \
            --max-num-batched-tokens 2048 \
            --kv-cache-memory 21653924864 \
            --dtype float16 \
            --kv-cache-dtype auto \
            --trust-remote-code \
            --enable-chunked-prefill \
            --no-enable-prefix-caching \
            --reasoning-parser qwen3 \
            --additional-config "{\"ascend_compilation_config\": {\"fuse_norm_quant\": false}}" \
            --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [1, 2, 4, 8]}" \
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
