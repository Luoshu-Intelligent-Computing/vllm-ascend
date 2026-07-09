#!/usr/bin/env bash
# start-128k-service.sh (nightly)
#
# 启动 Qwen3.6-35B-A3B-w8a8 128K 服务（nightly-main-310p 镜像）
#
# 与 poc 版的区别：
#   1. 使用 nightly 镜像（CANN 9.1.0-beta.1），GDN Prefill AscendC kernel 可用
#   2. 启动时重新编译 vllm_ascend_C.so（SOC_VERSION=ascend310p1 确保 ASCEND_PLATFORM_310P 生效）
#   3. 只需 attention_mask.py + metadata_builder.py 两个 patch（attention_v1.py 无需 patch）
#   4. 性能更好：Prefill 吞吐 ~700 t/s（vs poc 的 ~633 t/s）
#
# 用法:
#   ./start-128k-service.sh              # 启动（含编译，~30min 首次）
#   ./start-128k-service.sh --detach     # 后台启动
#   ./start-128k-service.sh --stop       # 停止
#   ./start-128k-service.sh --status     # 查看状态
#   ./start-128k-service.sh --logs       # 查看日志

set -e

CONTAINER_NAME="vllm-nightly-128k"
IMAGE="quay.io/ascend/vllm-ascend:nightly-main-310p"
PORT=18082
SERVED_MODEL_NAME="qwen3.6-128k-nightly"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/../src"
MODELS_DIR="/srv/meetai/models"

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

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_ready() {
    local start elapsed
    start=$(date +%s)
    log "等待服务就绪（含编译，首次约30分钟）..."
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
        if [ $elapsed -gt 3600 ]; then
            log "❌ 超时（3600s），请检查日志: sudo podman logs ${CONTAINER_NAME}"
            return 1
        fi
        last_log=$(sudo podman logs --tail 1 "${CONTAINER_NAME}" 2>&1 | tail -1 | cut -c1-100)
        echo "  [${elapsed}s] ${last_log}"
        sleep 30
    done
}

if [ "$ACTION" = "help" ]; then
    sed -n '2,15p' "$0" | sed 's/^# \?//'
    exit 0
fi

if [ "$ACTION" = "status" ]; then
    sudo podman ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    curl -sf "http://localhost:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || echo "（服务未响应）"
    exit 0
fi

if [ "$ACTION" = "logs" ]; then
    exec sudo podman logs -f "${CONTAINER_NAME}"
fi

if [ "$ACTION" = "stop" ]; then
    log "停止容器 ${CONTAINER_NAME}..."
    sudo podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    sudo podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
    log "✅ 已停止"
    exit 0
fi

# 检查 patch 文件
for f in metadata_builder.py attention_mask.py; do
    if [ ! -f "${PATCHES_DIR}/${f}" ]; then
        log "❌ 缺少 patch 文件: ${PATCHES_DIR}/${f}"
        exit 1
    fi
done

if sudo podman ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "发现已有容器，停止并清理..."
    sudo podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    sudo podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
fi

log "启动容器 ${CONTAINER_NAME}（nightly + GDN AscendC kernel）"

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
        source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh
        export ASCEND_CUSTOM_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom

        # 应用 patch（nightly 只需2个文件，attention_v1.py 无需 patch）
        VLLM_PATH=$(python3 -c "import vllm_ascend; import os; print(os.path.dirname(vllm_ascend.__file__))")
        cp /workspace/patches/attention_mask.py  $VLLM_PATH/_310p/attention/
        cp /workspace/patches/metadata_builder.py $VLLM_PATH/_310p/attention/
        echo "[INFO] Patches applied (nightly: 2 files)"

        # 重新编译确保 ASCEND_PLATFORM_310P 生效（注册 GDN chunk 算子）
        cd /vllm-workspace/vllm-ascend
        SOC_VERSION=ascend310p1 pip install -e . --no-build-isolation --no-deps -q
        echo "[INFO] Compilation done"

        exec python3 -m vllm.entrypoints.openai.api_server \
            --model /models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8 \
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

if [ "$ACTION" = "start" ]; then
    wait_ready
else
    log "后台模式，查看日志: $0 --logs"
fi
