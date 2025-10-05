#!/bin/bash

# Firstrade交易系统启动脚本
# 用于快速启动系统的各个组件

set -e

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

# 显示帮助信息
show_help() {
    echo "Firstrade交易系统启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -d, --dev           开发模式启动"
    echo "  -p, --prod          生产模式启动"
    echo "  -m, --monitoring    启动监控服务"
    echo "  -w, --web           启动Web界面"
    echo "  -a, --all           启动所有服务"
    echo "  --check             检查系统状态"
    echo ""
    echo "示例:"
    echo "  $0 --dev            # 开发模式启动"
    echo "  $0 --prod           # 生产模式启动"
    echo "  $0 --all            # 启动所有服务"
    echo "  $0 --check          # 检查系统状态"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker 未安装，将使用本地模式"
        USE_DOCKER=false
    else
        USE_DOCKER=true
    fi
    
    # 检查Redis
    if ! command -v redis-server &> /dev/null && [ "$USE_DOCKER" = false ]; then
        log_warning "Redis 未安装，某些功能可能不可用"
    fi
    
    log_success "依赖检查完成"
}

# 检查系统状态
check_system_status() {
    log_info "检查系统状态..."
    
    # 检查Python进程
    if pgrep -f "firstrade_trading_system" > /dev/null; then
        log_success "交易系统正在运行"
    else
        log_warning "交易系统未运行"
    fi
    
    # 检查监控服务
    if pgrep -f "monitoring_dashboard" > /dev/null; then
        log_success "监控服务正在运行"
    else
        log_warning "监控服务未运行"
    fi
    
    # 检查Web服务
    if pgrep -f "streamlit" > /dev/null; then
        log_success "Web界面正在运行"
    else
        log_warning "Web界面未运行"
    fi
    
    # 检查Docker服务
    if [ "$USE_DOCKER" = true ]; then
        if docker ps | grep -q "firstrade"; then
            log_success "Docker服务正在运行"
        else
            log_warning "Docker服务未运行"
        fi
    fi
}

# 启动核心交易系统
start_trading_system() {
    log_info "启动核心交易系统..."
    
    if [ "$MODE" = "dev" ]; then
        # 开发模式
        cd src
        python3 -m firstrade_trading_system --config ../config/config.yaml --debug &
        TRADING_PID=$!
        echo $TRADING_PID > ../pids/trading_system.pid
    else
        # 生产模式
        cd src
        python3 -m firstrade_trading_system --config ../config/config.yaml &
        TRADING_PID=$!
        echo $TRADING_PID > ../pids/trading_system.pid
    fi
    
    log_success "交易系统已启动 (PID: $TRADING_PID)"
}

# 启动监控服务
start_monitoring() {
    log_info "启动监控服务..."
    
    cd examples
    python3 monitoring_dashboard.py &
    MONITORING_PID=$!
    echo $MONITORING_PID > ../pids/monitoring.pid
    
    log_success "监控服务已启动 (PID: $MONITORING_PID)"
}

# 启动Web界面
start_web_interface() {
    log_info "启动Web界面..."
    
    if command -v streamlit &> /dev/null; then
        cd examples
        streamlit run web_interface.py --server.port 8501 &
        WEB_PID=$!
        echo $WEB_PID > ../pids/web_interface.pid
        log_success "Web界面已启动 (PID: $WEB_PID)"
        log_info "访问地址: http://localhost:8501"
    else
        log_warning "Streamlit 未安装，跳过Web界面启动"
    fi
}

# 启动Docker服务
start_docker_services() {
    log_info "启动Docker服务..."
    
    if [ "$USE_DOCKER" = true ]; then
        docker-compose up -d
        log_success "Docker服务已启动"
    else
        log_warning "Docker 不可用，跳过Docker服务启动"
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p pids
    mkdir -p logs
    mkdir -p data_cache
    mkdir -p backups
    mkdir -p results
    
    log_success "目录创建完成"
}

# 等待服务启动
wait_for_services() {
    log_info "等待服务启动..."
    sleep 5
    log_success "服务启动完成"
}

# 显示启动信息
show_startup_info() {
    echo ""
    log_success "=== Firstrade交易系统启动完成 ==="
    echo ""
    echo "服务信息:"
    
    if [ -f "pids/trading_system.pid" ]; then
        echo "  • 交易系统: 运行中 (PID: $(cat pids/trading_system.pid))"
    fi
    
    if [ -f "pids/monitoring.pid" ]; then
        echo "  • 监控服务: 运行中 (PID: $(cat pids/monitoring.pid))"
        echo "    访问地址: http://localhost:8080"
    fi
    
    if [ -f "pids/web_interface.pid" ]; then
        echo "  • Web界面: 运行中 (PID: $(cat pids/web_interface.pid))"
        echo "    访问地址: http://localhost:8501"
    fi
    
    if [ "$USE_DOCKER" = true ]; then
        echo "  • Docker服务: 运行中"
        echo "    查看状态: docker-compose ps"
    fi
    
    echo ""
    echo "管理命令:"
    echo "  • 查看状态: $0 --check"
    echo "  • 停止服务: ./scripts/stop.sh"
    echo "  • 查看日志: tail -f logs/*.log"
    echo ""
}

# 主函数
main() {
    # 默认参数
    MODE="dev"
    START_MONITORING=false
    START_WEB=false
    START_ALL=false
    CHECK_STATUS=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dev)
                MODE="dev"
                shift
                ;;
            -p|--prod)
                MODE="prod"
                shift
                ;;
            -m|--monitoring)
                START_MONITORING=true
                shift
                ;;
            -w|--web)
                START_WEB=true
                shift
                ;;
            -a|--all)
                START_ALL=true
                shift
                ;;
            --check)
                CHECK_STATUS=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查依赖
    check_dependencies
    
    # 如果只是检查状态
    if [ "$CHECK_STATUS" = true ]; then
        check_system_status
        exit 0
    fi
    
    # 创建必要的目录
    create_directories
    
    log_info "启动模式: $MODE"
    
    # 启动核心交易系统
    start_trading_system
    
    # 根据参数启动其他服务
    if [ "$START_ALL" = true ]; then
        START_MONITORING=true
        START_WEB=true
        if [ "$USE_DOCKER" = true ]; then
            start_docker_services
        fi
    fi
    
    if [ "$START_MONITORING" = true ]; then
        start_monitoring
    fi
    
    if [ "$START_WEB" = true ]; then
        start_web_interface
    fi
    
    # 等待服务启动
    wait_for_services
    
    # 显示启动信息
    show_startup_info
}

# 执行主函数
main "$@"