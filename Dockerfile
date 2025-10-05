# Firstrade 交易系统 Docker 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    gcc \
    g++ \
    make \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # Chrome浏览器依赖
    wget \
    gnupg \
    unzip \
    chromium \
    chromium-driver \
    # TA-Lib依赖
    libta-lib-dev \
    ta-lib \
    && rm -rf /var/lib/apt/lists/*

# 设置Chrome环境变量
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 创建非root用户
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app

# 复制应用代码
COPY . .

# 设置权限
RUN chown -R trader:trader /app && \
    chmod +x scripts/*.sh

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/config /app/cache && \
    chown -R trader:trader /app/logs /app/data /app/config /app/cache

# 切换到非root用户
USER trader

# 暴露端口
EXPOSE 8080 5000 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动脚本
CMD ["python", "examples/monitoring_dashboard.py"]