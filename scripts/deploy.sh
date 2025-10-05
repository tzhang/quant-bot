#!/bin/bash

# Firstrade 交易系统部署脚本
# 作者: Trading System Team
# 版本: 1.0.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 未安装，请先安装 $1"
        exit 1
    fi
}

# 检查Docker和Docker Compose
check_dependencies() {
    log_info "检查依赖..."
    check_command docker
    check_command docker-compose
    log_success "依赖检查完成"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    directories=(
        "logs"
        "data"
        "config"
        "cache"
        "nginx"
        "monitoring"
        "scripts"
        "ssl"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done
    
    log_success "目录创建完成"
}

# 生成配置文件
generate_configs() {
    log_info "生成配置文件..."
    
    # 生成Nginx配置
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream trading_system {
        server trading-system:8080;
    }
    
    upstream api_server {
        server trading-system:5000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://trading_system;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/ {
            proxy_pass http://api_server/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /socket.io/ {
            proxy_pass http://trading_system;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF

    # 生成Prometheus配置
    mkdir -p monitoring
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'trading-system'
    static_configs:
      - targets: ['trading-system:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF

    # 生成数据库初始化脚本
    cat > scripts/init_db.sql << 'EOF'
-- 创建交易系统数据库表

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 账户表
CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    account_number VARCHAR(50) UNIQUE NOT NULL,
    account_type VARCHAR(20) NOT NULL,
    balance DECIMAL(15,2) DEFAULT 0.00,
    buying_power DECIMAL(15,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 持仓表
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    symbol VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(10,4) NOT NULL,
    market_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 订单表
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4),
    status VARCHAR(20) NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 交易记录表
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES orders(order_id),
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0.00,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 市场数据表
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    volume BIGINT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略表
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parameters JSONB,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略执行记录表
CREATE TABLE IF NOT EXISTS strategy_executions (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    symbol VARCHAR(10) NOT NULL,
    signal VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 风险指标表
CREATE TABLE IF NOT EXISTS risk_metrics (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    date DATE NOT NULL,
    var_95 DECIMAL(15,2),
    var_99 DECIMAL(15,2),
    max_drawdown DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    beta DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_positions_account_symbol ON positions(account_id, symbol);
CREATE INDEX IF NOT EXISTS idx_orders_account_status ON orders(account_id, status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_date ON trades(symbol, executed_at);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_strategy_executions_strategy_date ON strategy_executions(strategy_id, executed_at);

-- 插入默认数据
INSERT INTO users (username, email, password_hash) VALUES 
('admin', 'admin@trading.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Gm.F5e')
ON CONFLICT (username) DO NOTHING;

INSERT INTO strategies (name, description, parameters, is_active) VALUES 
('RSI策略', 'RSI超买超卖策略', '{"rsi_period": 14, "oversold": 30, "overbought": 70}', true),
('移动平均策略', '双移动平均线交叉策略', '{"short_ma": 5, "long_ma": 20}', true),
('MACD策略', 'MACD金叉死叉策略', '{"fast_period": 12, "slow_period": 26, "signal_period": 9}', false)
ON CONFLICT DO NOTHING;
EOF

    log_success "配置文件生成完成"
}

# 构建Docker镜像
build_images() {
    log_info "构建Docker镜像..."
    
    # 构建主应用镜像
    docker build -t firstrade-trading-system:latest .
    
    log_success "Docker镜像构建完成"
}

# 启动服务
start_services() {
    log_info "启动服务..."
    
    # 停止现有服务
    docker-compose down --remove-orphans
    
    # 启动所有服务
    docker-compose up -d
    
    log_success "服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    # 等待数据库就绪
    log_info "等待PostgreSQL就绪..."
    timeout=60
    while ! docker-compose exec -T postgres pg_isready -U trader -d trading_db > /dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            log_error "PostgreSQL启动超时"
            exit 1
        fi
    done
    log_success "PostgreSQL已就绪"
    
    # 等待Redis就绪
    log_info "等待Redis就绪..."
    timeout=30
    while ! docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            log_error "Redis启动超时"
            exit 1
        fi
    done
    log_success "Redis已就绪"
    
    # 等待主应用就绪
    log_info "等待交易系统就绪..."
    timeout=120
    while ! curl -f http://localhost:8080/health > /dev/null 2>&1; do
        sleep 5
        timeout=$((timeout - 5))
        if [ $timeout -le 0 ]; then
            log_error "交易系统启动超时"
            exit 1
        fi
    done
    log_success "交易系统已就绪"
}

# 运行健康检查
health_check() {
    log_info "运行健康检查..."
    
    # 检查各个服务状态
    services=("postgres" "redis" "trading-system" "nginx")
    
    for service in "${services[@]}"; do
        if docker-compose ps $service | grep -q "Up"; then
            log_success "$service 运行正常"
        else
            log_error "$service 运行异常"
            docker-compose logs $service
        fi
    done
    
    # 检查端口
    ports=("80" "8080" "5432" "6379")
    
    for port in "${ports[@]}"; do
        if netstat -tuln | grep -q ":$port "; then
            log_success "端口 $port 已开放"
        else
            log_warning "端口 $port 未开放"
        fi
    done
}

# 显示部署信息
show_deployment_info() {
    log_success "部署完成！"
    echo ""
    echo "=== 服务访问地址 ==="
    echo "🌐 监控面板: http://localhost:8080"
    echo "📊 Grafana: http://localhost:3000 (admin/admin123)"
    echo "📈 Prometheus: http://localhost:9090"
    echo "🔧 API服务: http://localhost:5000"
    echo ""
    echo "=== 数据库连接信息 ==="
    echo "🐘 PostgreSQL: localhost:5432"
    echo "   数据库: trading_db"
    echo "   用户名: trader"
    echo "   密码: password"
    echo ""
    echo "📦 Redis: localhost:6379"
    echo ""
    echo "=== 常用命令 ==="
    echo "查看日志: docker-compose logs -f [service_name]"
    echo "重启服务: docker-compose restart [service_name]"
    echo "停止服务: docker-compose down"
    echo "更新服务: ./scripts/deploy.sh"
    echo ""
}

# 主函数
main() {
    log_info "开始部署 Firstrade 交易系统..."
    
    # 检查参数
    case "${1:-deploy}" in
        "deploy")
            check_dependencies
            create_directories
            generate_configs
            build_images
            start_services
            wait_for_services
            health_check
            show_deployment_info
            ;;
        "start")
            start_services
            wait_for_services
            log_success "服务启动完成"
            ;;
        "stop")
            log_info "停止服务..."
            docker-compose down
            log_success "服务已停止"
            ;;
        "restart")
            log_info "重启服务..."
            docker-compose restart
            wait_for_services
            log_success "服务重启完成"
            ;;
        "logs")
            docker-compose logs -f
            ;;
        "status")
            docker-compose ps
            health_check
            ;;
        "clean")
            log_warning "清理所有数据和镜像..."
            read -p "确认删除所有数据？(y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                docker-compose down -v --remove-orphans
                docker system prune -f
                log_success "清理完成"
            else
                log_info "取消清理"
            fi
            ;;
        "help"|"-h"|"--help")
            echo "用法: $0 [命令]"
            echo ""
            echo "命令:"
            echo "  deploy   完整部署 (默认)"
            echo "  start    启动服务"
            echo "  stop     停止服务"
            echo "  restart  重启服务"
            echo "  logs     查看日志"
            echo "  status   查看状态"
            echo "  clean    清理数据"
            echo "  help     显示帮助"
            ;;
        *)
            log_error "未知命令: $1"
            echo "使用 '$0 help' 查看帮助"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"