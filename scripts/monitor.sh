#!/bin/bash

# Firstrade交易系统监控脚本
# 用于监控系统运行状态和性能指标

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置参数
MONITOR_INTERVAL=5
LOG_FILE="logs/monitor.log"
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=80
ALERT_THRESHOLD_DISK=90
ALERT_EMAIL=""
ALERT_WEBHOOK=""

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $1" >> "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "$LOG_FILE"
}

log_alert() {
    echo -e "${RED}[ALERT]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ALERT] $1" >> "$LOG_FILE"
    send_alert "$1"
}

# 显示帮助信息
show_help() {
    echo "Firstrade交易系统监控脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -i, --interval SECONDS  监控间隔（默认: 5秒）"
    echo "  -l, --log FILE          日志文件路径"
    echo "  -e, --email EMAIL       告警邮箱地址"
    echo "  -w, --webhook URL       告警Webhook URL"
    echo "  --cpu-threshold NUM     CPU使用率告警阈值（默认: 80%）"
    echo "  --memory-threshold NUM  内存使用率告警阈值（默认: 80%）"
    echo "  --disk-threshold NUM    磁盘使用率告警阈值（默认: 90%）"
    echo "  --status                显示当前状态"
    echo "  --report                生成监控报告"
    echo "  --dashboard             启动实时监控面板"
    echo ""
    echo "示例:"
    echo "  $0                      # 开始监控"
    echo "  $0 --interval 10        # 10秒间隔监控"
    echo "  $0 --status             # 显示当前状态"
    echo "  $0 --report             # 生成监控报告"
}

# 发送告警
send_alert() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 邮件告警
    if [ -n "$ALERT_EMAIL" ]; then
        if command -v mail &> /dev/null; then
            echo "时间: $timestamp" | mail -s "Firstrade系统告警: $message" "$ALERT_EMAIL"
        fi
    fi
    
    # Webhook告警
    if [ -n "$ALERT_WEBHOOK" ]; then
        if command -v curl &> /dev/null; then
            curl -X POST "$ALERT_WEBHOOK" \
                -H "Content-Type: application/json" \
                -d "{\"text\":\"Firstrade系统告警\\n时间: $timestamp\\n消息: $message\"}" \
                2>/dev/null || true
        fi
    fi
}

# 获取系统信息
get_system_info() {
    # CPU使用率
    if command -v top &> /dev/null; then
        CPU_USAGE=$(top -l 1 -s 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' 2>/dev/null || echo "0")
    else
        CPU_USAGE="0"
    fi
    
    # 内存使用率
    if command -v vm_stat &> /dev/null; then
        local vm_stat_output=$(vm_stat)
        local pages_free=$(echo "$vm_stat_output" | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        local pages_active=$(echo "$vm_stat_output" | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
        local pages_inactive=$(echo "$vm_stat_output" | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
        local pages_speculative=$(echo "$vm_stat_output" | grep "Pages speculative" | awk '{print $3}' | sed 's/\.//')
        local pages_wired=$(echo "$vm_stat_output" | grep "Pages wired down" | awk '{print $4}' | sed 's/\.//')
        
        local total_pages=$((pages_free + pages_active + pages_inactive + pages_speculative + pages_wired))
        local used_pages=$((pages_active + pages_inactive + pages_speculative + pages_wired))
        
        if [ $total_pages -gt 0 ]; then
            MEMORY_USAGE=$((used_pages * 100 / total_pages))
        else
            MEMORY_USAGE="0"
        fi
    else
        MEMORY_USAGE="0"
    fi
    
    # 磁盘使用率
    DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//' 2>/dev/null || echo "0")
    
    # 网络连接数
    NETWORK_CONNECTIONS=$(netstat -an 2>/dev/null | wc -l || echo "0")
    
    # 负载平均值
    if command -v uptime &> /dev/null; then
        LOAD_AVERAGE=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}' | sed 's/,//' 2>/dev/null || echo "0.00")
    else
        LOAD_AVERAGE="0.00"
    fi
}

# 检查进程状态
check_process_status() {
    # 交易系统状态
    if pgrep -f "firstrade_trading_system" > /dev/null; then
        TRADING_STATUS="运行中"
        TRADING_PID=$(pgrep -f "firstrade_trading_system")
    else
        TRADING_STATUS="已停止"
        TRADING_PID=""
    fi
    
    # 监控服务状态
    if pgrep -f "monitoring_dashboard" > /dev/null; then
        MONITORING_STATUS="运行中"
        MONITORING_PID=$(pgrep -f "monitoring_dashboard")
    else
        MONITORING_STATUS="已停止"
        MONITORING_PID=""
    fi
    
    # Web界面状态
    if pgrep -f "streamlit" > /dev/null; then
        WEB_STATUS="运行中"
        WEB_PID=$(pgrep -f "streamlit")
    else
        WEB_STATUS="已停止"
        WEB_PID=""
    fi
    
    # Docker服务状态
    if command -v docker &> /dev/null && docker ps | grep -q "firstrade"; then
        DOCKER_STATUS="运行中"
    else
        DOCKER_STATUS="已停止"
    fi
}

# 检查端口状态
check_port_status() {
    # 检查关键端口
    PORTS=("8080" "8501" "5432" "6379")
    PORT_STATUS=""
    
    for port in "${PORTS[@]}"; do
        if netstat -an | grep -q ":$port.*LISTEN"; then
            PORT_STATUS="$PORT_STATUS $port:开放"
        else
            PORT_STATUS="$PORT_STATUS $port:关闭"
        fi
    done
}

# 检查日志错误
check_log_errors() {
    local error_count=0
    
    # 检查最近的错误日志
    if [ -d "logs" ]; then
        error_count=$(find logs -name "*.log" -mtime -1 -exec grep -c "ERROR\|CRITICAL" {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
    fi
    
    LOG_ERRORS=$error_count
}

# 显示实时状态
show_status() {
    clear
    echo -e "${CYAN}=== Firstrade交易系统监控面板 ===${NC}"
    echo -e "${CYAN}更新时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
    
    # 系统资源
    echo -e "${BLUE}系统资源:${NC}"
    printf "  CPU使用率:    %s%%\n" "$CPU_USAGE"
    printf "  内存使用率:   %s%%\n" "$MEMORY_USAGE"
    printf "  磁盘使用率:   %s%%\n" "$DISK_USAGE"
    printf "  负载平均值:   %s\n" "$LOAD_AVERAGE"
    printf "  网络连接数:   %s\n" "$NETWORK_CONNECTIONS"
    echo ""
    
    # 服务状态
    echo -e "${BLUE}服务状态:${NC}"
    printf "  交易系统:     %s" "$TRADING_STATUS"
    [ -n "$TRADING_PID" ] && printf " (PID: %s)" "$TRADING_PID"
    echo ""
    printf "  监控服务:     %s" "$MONITORING_STATUS"
    [ -n "$MONITORING_PID" ] && printf " (PID: %s)" "$MONITORING_PID"
    echo ""
    printf "  Web界面:      %s" "$WEB_STATUS"
    [ -n "$WEB_PID" ] && printf " (PID: %s)" "$WEB_PID"
    echo ""
    printf "  Docker服务:   %s\n" "$DOCKER_STATUS"
    echo ""
    
    # 端口状态
    echo -e "${BLUE}端口状态:${NC}"
    echo "  $PORT_STATUS"
    echo ""
    
    # 日志错误
    echo -e "${BLUE}日志状态:${NC}"
    printf "  最近24小时错误数: %s\n" "$LOG_ERRORS"
    echo ""
    
    # 告警状态
    echo -e "${BLUE}告警状态:${NC}"
    if [ "${CPU_USAGE%.*}" -gt "$ALERT_THRESHOLD_CPU" ]; then
        echo -e "  ${RED}CPU使用率过高: ${CPU_USAGE}%${NC}"
    fi
    if [ "${MEMORY_USAGE%.*}" -gt "$ALERT_THRESHOLD_MEMORY" ]; then
        echo -e "  ${RED}内存使用率过高: ${MEMORY_USAGE}%${NC}"
    fi
    if [ "${DISK_USAGE%.*}" -gt "$ALERT_THRESHOLD_DISK" ]; then
        echo -e "  ${RED}磁盘使用率过高: ${DISK_USAGE}%${NC}"
    fi
    
    if [ "$TRADING_STATUS" = "已停止" ]; then
        echo -e "  ${RED}交易系统未运行${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}按 Ctrl+C 退出监控${NC}"
}

# 生成监控报告
generate_report() {
    local report_file="logs/monitor_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "生成监控报告..."
    
    {
        echo "=== Firstrade交易系统监控报告 ==="
        echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        echo "系统资源使用情况:"
        echo "  CPU使用率: ${CPU_USAGE}%"
        echo "  内存使用率: ${MEMORY_USAGE}%"
        echo "  磁盘使用率: ${DISK_USAGE}%"
        echo "  负载平均值: ${LOAD_AVERAGE}"
        echo "  网络连接数: ${NETWORK_CONNECTIONS}"
        echo ""
        
        echo "服务运行状态:"
        echo "  交易系统: $TRADING_STATUS"
        echo "  监控服务: $MONITORING_STATUS"
        echo "  Web界面: $WEB_STATUS"
        echo "  Docker服务: $DOCKER_STATUS"
        echo ""
        
        echo "端口监听状态:"
        echo "  $PORT_STATUS"
        echo ""
        
        echo "日志统计:"
        echo "  最近24小时错误数: $LOG_ERRORS"
        echo ""
        
        echo "系统进程:"
        ps aux | grep -E "(firstrade|monitoring|streamlit)" | grep -v grep || echo "  无相关进程"
        echo ""
        
        echo "网络连接:"
        netstat -an | grep -E "(8080|8501|5432|6379)" || echo "  无相关连接"
        echo ""
        
        echo "磁盘使用详情:"
        df -h
        echo ""
        
        echo "内存使用详情:"
        if command -v free &> /dev/null; then
            free -h
        else
            vm_stat
        fi
        
    } > "$report_file"
    
    log_success "监控报告已生成: $report_file"
}

# 检查告警条件
check_alerts() {
    # CPU告警
    if [ "${CPU_USAGE%.*}" -gt "$ALERT_THRESHOLD_CPU" ]; then
        log_alert "CPU使用率过高: ${CPU_USAGE}%"
    fi
    
    # 内存告警
    if [ "${MEMORY_USAGE%.*}" -gt "$ALERT_THRESHOLD_MEMORY" ]; then
        log_alert "内存使用率过高: ${MEMORY_USAGE}%"
    fi
    
    # 磁盘告警
    if [ "${DISK_USAGE%.*}" -gt "$ALERT_THRESHOLD_DISK" ]; then
        log_alert "磁盘使用率过高: ${DISK_USAGE}%"
    fi
    
    # 服务状态告警
    if [ "$TRADING_STATUS" = "已停止" ]; then
        log_alert "交易系统未运行"
    fi
    
    # 日志错误告警
    if [ "$LOG_ERRORS" -gt 10 ]; then
        log_alert "最近24小时错误数过多: $LOG_ERRORS"
    fi
}

# 启动实时监控面板
start_dashboard() {
    log_info "启动实时监控面板..."
    
    # 创建日志目录
    mkdir -p logs
    
    # 捕获退出信号
    trap 'log_info "监控面板已停止"; exit 0' INT TERM
    
    while true; do
        # 获取系统信息
        get_system_info
        check_process_status
        check_port_status
        check_log_errors
        
        # 显示状态
        show_status
        
        # 检查告警
        check_alerts
        
        # 等待下一次更新
        sleep "$MONITOR_INTERVAL"
    done
}

# 主函数
main() {
    # 默认参数
    SHOW_STATUS=false
    GENERATE_REPORT=false
    START_DASHBOARD=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -i|--interval)
                MONITOR_INTERVAL="$2"
                shift 2
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            -e|--email)
                ALERT_EMAIL="$2"
                shift 2
                ;;
            -w|--webhook)
                ALERT_WEBHOOK="$2"
                shift 2
                ;;
            --cpu-threshold)
                ALERT_THRESHOLD_CPU="$2"
                shift 2
                ;;
            --memory-threshold)
                ALERT_THRESHOLD_MEMORY="$2"
                shift 2
                ;;
            --disk-threshold)
                ALERT_THRESHOLD_DISK="$2"
                shift 2
                ;;
            --status)
                SHOW_STATUS=true
                shift
                ;;
            --report)
                GENERATE_REPORT=true
                shift
                ;;
            --dashboard)
                START_DASHBOARD=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 创建日志目录
    mkdir -p logs
    
    # 获取系统信息
    get_system_info
    check_process_status
    check_port_status
    check_log_errors
    
    # 根据参数执行相应操作
    if [ "$SHOW_STATUS" = true ]; then
        show_status
    elif [ "$GENERATE_REPORT" = true ]; then
        generate_report
    elif [ "$START_DASHBOARD" = true ]; then
        start_dashboard
    else
        # 默认启动监控面板
        start_dashboard
    fi
}

# 执行主函数
main "$@"