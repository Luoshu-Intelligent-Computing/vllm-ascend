#!/usr/bin/env bash
# 拉取 openEuler 310P 基础镜像
# 针对网络不稳定的情况提供多种备选方案

set -e

IMAGE="quay.io/ascend/cann:9.1.0-beta.1-310p-openeuler24.03-py3.12"

echo "=========================================="
echo "拉取 openEuler 310P 基础镜像"
echo "镜像: ${IMAGE}"
echo "=========================================="
echo ""

# ============================================================================
# 方案 1: 使用代理拉取（推荐）
# ============================================================================
echo "【方案 1】使用代理拉取（端口 10000）"
echo "命令: https_proxy=http://127.0.0.1:10000 sudo podman pull ${IMAGE}"
echo ""
read -p "尝试方案 1? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始拉取..."
    https_proxy=http://127.0.0.1:10000 sudo podman pull "${IMAGE}" && {
        echo "✅ 拉取成功"
        exit 0
    } || echo "❌ 拉取失败，尝试其他方案"
fi

# ============================================================================
# 方案 2: 直连拉取（无代理）
# ============================================================================
echo ""
echo "【方案 2】直连拉取（无代理）"
echo "命令: sudo podman pull ${IMAGE}"
echo ""
read -p "尝试方案 2? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始拉取..."
    sudo podman pull "${IMAGE}" && {
        echo "✅ 拉取成功"
        exit 0
    } || echo "❌ 拉取失败，尝试其他方案"
fi

# ============================================================================
# 方案 3: 分块拉取（适用于大镜像网络超时）
# ============================================================================
echo ""
echo "【方案 3】分块拉取（设置超时参数）"
echo "命令: sudo podman pull --retry 5 --retry-delay 10s ${IMAGE}"
echo ""
read -p "尝试方案 3? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始拉取..."
    https_proxy=http://127.0.0.1:10000 \
    sudo podman pull --retry 5 --retry-delay 10s "${IMAGE}" && {
        echo "✅ 拉取成功"
        exit 0
    } || echo "❌ 拉取失败，尝试其他方案"
fi

# ============================================================================
# 方案 4: 使用 skopeo 复制（绕过 podman 的网络问题）
# ============================================================================
echo ""
echo "【方案 4】使用 skopeo 工具"
echo "先安装: sudo apt install skopeo"
echo "命令: skopeo copy --override-os linux --override-arch arm64 \\"
echo "       docker://${IMAGE} \\"
echo "       containers-storage:${IMAGE}"
echo ""
read -p "尝试方案 4? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ! command -v skopeo &> /dev/null; then
        echo "安装 skopeo..."
        sudo apt update && sudo apt install -y skopeo
    fi
    echo "开始拉取..."
    https_proxy=http://127.0.0.1:10000 \
    skopeo copy \
        --override-os linux \
        --override-arch arm64 \
        "docker://${IMAGE}" \
        "containers-storage:${IMAGE}" && {
        echo "✅ 拉取成功"
        exit 0
    } || echo "❌ 拉取失败"
fi

# ============================================================================
# 方案 5: 手动命令（用户自行调试）
# ============================================================================
echo ""
echo "=========================================="
echo "所有自动方案均失败，手动调试建议："
echo "=========================================="
echo ""
echo "1. 检查代理状态："
echo "   curl -x http://127.0.0.1:10000 -I https://quay.io"
echo ""
echo "2. 检查网络连接："
echo "   ping -c 3 quay.io"
echo ""
echo "3. 查看 podman 详细日志："
echo "   https_proxy=http://127.0.0.1:10000 sudo podman pull --log-level=debug ${IMAGE}"
echo ""
echo "4. 尝试其他代理端口："
echo "   https_proxy=http://127.0.0.1:9999 sudo podman pull ${IMAGE}"
echo ""
echo "5. 检查磁盘空间："
echo "   df -h /var/lib/containers"
echo ""
echo "6. 清理镜像缓存后重试："
echo "   sudo podman system prune -a -f"
echo "   https_proxy=http://127.0.0.1:10000 sudo podman pull ${IMAGE}"
echo ""
echo "7. 如果长时间卡住，考虑使用 tmux 后台运行："
echo "   tmux new -s pull-image"
echo "   https_proxy=http://127.0.0.1:10000 sudo podman pull ${IMAGE}"
echo "   # Ctrl+B, D 分离会话"
echo "   # tmux attach -t pull-image  # 重新连接"
echo ""
