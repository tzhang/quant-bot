#!/bin/bash

# Firstrade交易系统停止脚本
# 用于安全停止系统的各个组件

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
    echo "Firstrade交易系统停止脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -f, --force         强制停止所有服务"
    echo "  -g, --graceful      优雅停止（默认）"
    echo "  -d, --docker        停止Docker服务"
    echo "  -a, --all           停止所有服务"
    echo "  --trading           仅停止交易系统"
    echo "  --monitoring        仅停止监控服务"
    echo "  --web               仅停止Web界面"
    echo ""
    echo "示例:"
    echo "  $0                  # 优雅停止所有服务"
    echo "  $0 --force          # 强制停止所有服务"
    echo "  $0 --trading        # 仅停止交易系统"
    echo "  $0 --docker         # 停止Docker服务"
}

# 停止进程函数
stop_process() {
    local pid_file=$1
    local service_name=$2
    local force=$3
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            log_info "停止 $service_name (PID: $pid)..."
            
            if [ "$force" = true ]; then
                kill -9 $pid
            else
                kill -TERM $pid
                # 等待进程优雅退出
                local count=0
                while ps -p $pid > /dev/null 2>&1 && [ $count -lt 30 ]; do
                    sleep 1
                    count=$((count + 1))
                done
                
                # 如果进程仍在运行，强制终止
                if ps -p $pid > /dev/null 2>&1; then
                    log_warning "$service_name 未能优雅退出，强制终止..."
                    kill -9 $pid
                fi
            fi
            
            log_success "$service_name 已停止"
        else
            log_warning "$service_name 进程不存在 (PID: $pid)"
        fi
        rm -f "$pid_file"
    else
        log_warning "$service_name PID文件不存在"
    fi
}

# 停止交易系统
stop_trading_system() {
    log_info "停止交易系统..."
    stop_process "pids/trading_system.pid" "交易系统" $FORCE_STOP
    
    # 额外检查是否有遗留的交易系统进程
    local pids=$(pgrep -f "firstrade_trading_system" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log_info "发现遗留的交易系统进程，正在清理..."
        for pid in $pids; do
            if [ "$FORCE_STOP" = true ]; then
                kill -9 $pid
            else
                kill -TERM $pid
            fi
        done
        log_success "遗留进程已清理"
    fi
}

# 停止监控服务
stop_monitoring() {
    log_info "停止监控服务..."
    stop_process "pids/monitoring.pid" "监控服务" $FORCE_STOP
    
    # 额外检查监控相关进程
    local pids=$(pgrep -f "monitoring_dashboard" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log_info "发现遗留的监控进程，正在清理..."
        for pid in $pids; do
            if [ "$FORCE_STOP" = true ]; then
                kill -9 $pid
            else
                kill -TERM $pid
            fi
        done
        log_success "监控进程已清理"
    fi
}

# 停止Web界面
stop_web_interface() {
    log_info "停止Web界面..."
    stop_process "pids/web_interface.pid" "Web界面" $FORCE_STOP
    
    # 额外检查Streamlit进程
    local pids=$(pgrep -f "streamlit" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log_info "发现遗留的Streamlit进程，正在清理..."
        for pid in $pids; do
            if [ "$FORCE_STOP" = true ]; then
                kill -9 $pid
            else
                kill -TERM $pid
            fi
        done
        log_success "Streamlit进程已清理"
    fi
}

# 停止Docker服务
stop_docker_services() {
    log_info "停止Docker服务..."
    
    if command -v docker-compose &> /dev/null; then
        if [ -f "docker-compose.yml" ]; then
            docker-compose down
            log_success "Docker服务已停止"
        else
            log_warning "docker-compose.yml 文件不存在"
        fi
    else
        log_warning "docker-compose 未安装"
    fi
}

# 清理临时文件
cleanup_temp_files() {
    log_info "清理临时文件..."
    
    # 清理PID文件
    rm -f pids/*.pid
    
    # 清理临时日志文件
    find logs -name "*.tmp" -delete 2>/dev/null || true
    
    # 清理缓存文件
    find data_cache -name "*.tmp" -delete 2>/dev/null || true
    
    log_success "临时文件清理完成"
}

# 保存系统状态
save_system_state() {
    log_info "保存系统状态..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local state_file="logs/system_state_${timestamp}.log"
    
    {
        echo "=== 系统停止时间: $(date) ==="
        echo ""
        echo "运行中的进程:"
        ps aux | grep -E "(firstrade|monitoring|streamlit)" | grep -v grep || echo "无相关进程"
        echo ""
        echo "网络连接:"
        netstat -an | grep -E "(8080|8501|5432|6379)" || echo "无相关连接"
        echo ""
        echo "磁盘使用情况:"
        df -h
        echo ""
        echo "内存使用情况:"
        free -h 2>/dev/null || vm_stat
    } > "$state_file"
    
    log_success "系统状态已保存到: $state_file"
}

# 检查是否有进程仍在运行
check_remaining_processes() {
    log_info "检查剩余进程..."
    
    local remaining_pids=$(pgrep -f "(firstrade|monitoring|streamlit)" 2>/dev/null || true)
    if [ -n "$remaining_pids" ]; then
        log_warning "发现以下进程仍在运行:"
        for pid in $remaining_pids; do
            ps -p $pid -o pid,ppid,cmd 2>/dev/null || true
        done
        
        if [ "$FORCE_STOP" = true ]; then
            log_info "强制终止剩余进程..."
            for pid in $remaining_pids; do
                kill -9 $pid 2>/dev/null || true
            done
            log_success "剩余进程已强制终止"
        else
            log_warning "使用 --force 参数强制终止剩余进程"
        fi
    else
        log_success "所有相关进程已停止"
    fi
}

# 显示停止信息
show_stop_info() {
    echo ""
    log_success "=== Firstrade交易系统停止完成 ==="
    echo ""
    echo "停止的服务:"
    
    if [ "$STOP_TRADING" = true ] || [ "$STOP_ALL" = true ]; then
        echo "  • 交易系统: 已停止"
    fi
    
    if [ "$STOP_MONITORING" = true ] || [ "$STOP_ALL" = true ]; then
        echo "  • 监控服务: 已停止"
    fi
    
    if [ "$STOP_WEB" = true ] || [ "$STOP_ALL" = true ]; then
        echo "  • Web界面: 已停止"
    fi
    
    if [ "$STOP_DOCKER" = true ] || [ "$STOP_ALL" = true ]; then
        echo "  • Docker服务: 已停止"
    fi
    
    echo ""
    echo "管理命令:"
    echo "  • 重新启动: ./scripts/start.sh"
    echo "  • 查看日志: tail -f logs/*.log"
    echo "  • 清理数据: ./scripts/clean.sh"
    echo ""
}

# 主函数
main() {
    # 默认参数
    FORCE_STOP=false
    GRACEFUL_STOP=true
    STOP_ALL=true
    STOP_TRADING=false
    STOP_MONITORING=false
    STOP_WEB=false
    STOP_DOCKER=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -f|--force)
                FORCE_STOP=true
                GRACEFUL_STOP=false
                shift
                ;;
            -g|--graceful)
                GRACEFUL_STOP=true
                FORCE_STOP=false
                shift
                ;;
            -d|--docker)
                STOP_DOCKER=true
                STOP_ALL=false
                shift
                ;;
            -a|--all)
                STOP_ALL=true
                shift
                ;;
            --trading)
                STOP_TRADING=true
                STOP_ALL=false
                shift
                ;;
            --monitoring)
                STOP_MONITORING=true
                STOP_ALL=false
                shift
                ;;
            --web)
                STOP_WEB=true
                STOP_ALL=false
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log_info "开始停止Firstrade交易系统..."
    
    if [ "$FORCE_STOP" = true ]; then
        log_warning "使用强制停止模式"
    else
        log_info "使用优雅停止模式"
    fi
    
    # 保存系统状态
    save_system_state
    
    # 根据参数停止相应服务
    if [ "$STOP_ALL" = true ]; then
        stop_trading_system
        stop_monitoring
        stop_web_interface
        stop_docker_services
    else
        if [ "$STOP_TRADING" = true ]; then
            stop_trading_system
        fi
        
        if [ "$STOP_MONITORING" = true ]; then
            stop_monitoring
        fi
        
        if [ "$STOP_WEB" = true ]; then
            stop_web_interface
        fi
        
        if [ "$STOP_DOCKER" = true ]; then
            stop_docker_services
        fi
    fi
    
    # 清理临时文件
    cleanup_temp_files
    
    # 检查剩余进程
    check_remaining_processes
    
    # 显示停止信息
    show_stop_info
}

# 执行主函数
main "$@"