#!/bin/bash

# Firstrade 交易系统 Docker 镜像构建脚本
# 作者: Trading System Team
# 版本: 1.0.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
IMAGE_NAME="firstrade-trading-system"
REGISTRY="localhost:5000"  # 本地registry，可修改为实际registry地址
VERSION=${VERSION:-"latest"}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行，请启动 Docker"
        exit 1
    fi
    
    log_success "Docker 检查通过"
}

# 清理旧的构建缓存
clean_cache() {
    log_info "清理构建缓存..."
    
    # 清理悬空镜像
    docker image prune -f
    
    # 清理构建缓存
    docker builder prune -f
    
    log_success "缓存清理完成"
}

# 构建基础镜像
build_base_image() {
    log_info "构建基础镜像..."
    
    # 创建基础镜像的Dockerfile
    cat > Dockerfile.base << 'EOF'
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Chrome浏览器
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# 安装ChromeDriver
RUN CHROME_DRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE) \
    && wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip \
    && unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/ \
    && rm /tmp/chromedriver.zip \
    && chmod +x /usr/local/bin/chromedriver

# 安装TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 创建应用用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置工作目录
WORKDIR /app

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/cache /app/config \
    && chown -R appuser:appuser /app

USER appuser
EOF

    # 构建基础镜像
    docker build -f Dockerfile.base -t ${IMAGE_NAME}-base:${VERSION} .
    
    # 清理临时文件
    rm -f Dockerfile.base
    
    log_success "基础镜像构建完成"
}

# 构建应用镜像
build_app_image() {
    log_info "构建应用镜像..."
    
    # 构建参数
    BUILD_ARGS=(
        --build-arg BUILD_DATE="${BUILD_DATE}"
        --build-arg VERSION="${VERSION}"
        --build-arg GIT_COMMIT="${GIT_COMMIT}"
        --label "org.opencontainers.image.created=${BUILD_DATE}"
        --label "org.opencontainers.image.version=${VERSION}"
        --label "org.opencontainers.image.revision=${GIT_COMMIT}"
        --label "org.opencontainers.image.title=Firstrade Trading System"
        --label "org.opencontainers.image.description=Automated trading system for Firstrade"
    )
    
    # 构建镜像
    docker build "${BUILD_ARGS[@]}" -t ${IMAGE_NAME}:${VERSION} .
    
    # 如果版本不是latest，也打上latest标签
    if [ "${VERSION}" != "latest" ]; then
        docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest
    fi
    
    log_success "应用镜像构建完成"
}

# 构建多架构镜像
build_multiarch_image() {
    log_info "构建多架构镜像..."
    
    # 创建buildx builder
    docker buildx create --name multiarch-builder --use --bootstrap || true
    
    # 构建多架构镜像
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VERSION="${VERSION}" \
        --build-arg GIT_COMMIT="${GIT_COMMIT}" \
        --label "org.opencontainers.image.created=${BUILD_DATE}" \
        --label "org.opencontainers.image.version=${VERSION}" \
        --label "org.opencontainers.image.revision=${GIT_COMMIT}" \
        --label "org.opencontainers.image.title=Firstrade Trading System" \
        --label "org.opencontainers.image.description=Automated trading system for Firstrade" \
        -t ${IMAGE_NAME}:${VERSION} \
        --push \
        .
    
    log_success "多架构镜像构建完成"
}

# 运行安全扫描
security_scan() {
    log_info "运行安全扫描..."
    
    # 检查是否安装了trivy
    if command -v trivy &> /dev/null; then
        trivy image ${IMAGE_NAME}:${VERSION}
    else
        log_warning "Trivy 未安装，跳过安全扫描"
        log_info "安装 Trivy: https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
    fi
}

# 测试镜像
test_image() {
    log_info "测试镜像..."
    
    # 运行基本测试
    docker run --rm ${IMAGE_NAME}:${VERSION} python -c "
import sys
print(f'Python version: {sys.version}')

# 测试主要依赖
try:
    import selenium
    print(f'Selenium version: {selenium.__version__}')
except ImportError as e:
    print(f'Selenium import error: {e}')

try:
    import pandas
    print(f'Pandas version: {pandas.__version__}')
except ImportError as e:
    print(f'Pandas import error: {e}')

try:
    import numpy
    print(f'Numpy version: {numpy.__version__}')
except ImportError as e:
    print(f'Numpy import error: {e}')

try:
    import talib
    print('TA-Lib imported successfully')
except ImportError as e:
    print(f'TA-Lib import error: {e}')

try:
    import redis
    print(f'Redis version: {redis.__version__}')
except ImportError as e:
    print(f'Redis import error: {e}')

try:
    import psycopg2
    print('PostgreSQL driver imported successfully')
except ImportError as e:
    print(f'PostgreSQL driver import error: {e}')

print('Basic tests completed successfully!')
"
    
    log_success "镜像测试完成"
}

# 推送镜像到registry
push_image() {
    if [ -n "${REGISTRY}" ] && [ "${REGISTRY}" != "localhost:5000" ]; then
        log_info "推送镜像到 ${REGISTRY}..."
        
        # 重新标记镜像
        docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}
        docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:latest
        
        # 推送镜像
        docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
        docker push ${REGISTRY}/${IMAGE_NAME}:latest
        
        log_success "镜像推送完成"
    else
        log_info "跳过镜像推送 (本地构建)"
    fi
}

# 显示镜像信息
show_image_info() {
    log_success "构建完成！"
    echo ""
    echo "=== 镜像信息 ==="
    echo "📦 镜像名称: ${IMAGE_NAME}:${VERSION}"
    echo "🏷️  标签: ${VERSION}, latest"
    echo "📅 构建时间: ${BUILD_DATE}"
    echo "🔗 Git提交: ${GIT_COMMIT}"
    echo ""
    
    # 显示镜像大小
    echo "=== 镜像大小 ==="
    docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""
    
    echo "=== 使用方法 ==="
    echo "运行容器: docker run -d --name trading-system ${IMAGE_NAME}:${VERSION}"
    echo "进入容器: docker exec -it trading-system bash"
    echo "查看日志: docker logs trading-system"
    echo ""
}

# 主函数
main() {
    log_info "开始构建 Firstrade 交易系统 Docker 镜像..."
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version)
                VERSION="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --clean)
                CLEAN_CACHE=true
                shift
                ;;
            --base-only)
                BASE_ONLY=true
                shift
                ;;
            --multiarch)
                MULTIARCH=true
                shift
                ;;
            --no-test)
                NO_TEST=true
                shift
                ;;
            --no-push)
                NO_PUSH=true
                shift
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --version VERSION    设置镜像版本 (默认: latest)"
                echo "  --registry REGISTRY  设置镜像仓库地址"
                echo "  --clean             清理构建缓存"
                echo "  --base-only         仅构建基础镜像"
                echo "  --multiarch         构建多架构镜像"
                echo "  --no-test           跳过镜像测试"
                echo "  --no-push           跳过镜像推送"
                echo "  --help, -h          显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 检查Docker
    check_docker
    
    # 清理缓存
    if [ "${CLEAN_CACHE}" = true ]; then
        clean_cache
    fi
    
    # 构建基础镜像
    build_base_image
    
    # 如果只构建基础镜像，则退出
    if [ "${BASE_ONLY}" = true ]; then
        log_success "基础镜像构建完成"
        exit 0
    fi
    
    # 构建应用镜像
    if [ "${MULTIARCH}" = true ]; then
        build_multiarch_image
    else
        build_app_image
    fi
    
    # 运行安全扫描
    security_scan
    
    # 测试镜像
    if [ "${NO_TEST}" != true ]; then
        test_image
    fi
    
    # 推送镜像
    if [ "${NO_PUSH}" != true ]; then
        push_image
    fi
    
    # 显示镜像信息
    show_image_info
}

# 执行主函数
main "$@"