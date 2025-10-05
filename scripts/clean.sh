#!/bin/bash

# Firstrade交易系统清理脚本
# 用于清理系统缓存、日志和临时文件

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
    echo "Firstrade交易系统清理脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -a, --all           清理所有内容"
    echo "  -l, --logs          清理日志文件"
    echo "  -c, --cache         清理缓存文件"
    echo "  -t, --temp          清理临时文件"
    echo "  -b, --backups       清理旧备份文件"
    echo "  -d, --docker        清理Docker资源"
    echo "  -p, --pycache       清理Python缓存"
    echo "  --dry-run           预览清理操作（不实际删除）"
    echo "  --keep-days DAYS    保留最近N天的文件（默认: 7）"
    echo "  --force             强制清理（不询问确认）"
    echo ""
    echo "示例:"
    echo "  $0 --all            # 清理所有内容"
    echo "  $0 --logs --cache   # 仅清理日志和缓存"
    echo "  $0 --dry-run        # 预览清理操作"
    echo "  $0 --keep-days 3    # 保留最近3天的文件"
}

# 获取文件大小（人类可读格式）
get_size() {
    local path="$1"
    if [ -d "$path" ]; then
        du -sh "$path" 2>/dev/null | cut -f1 || echo "0B"
    elif [ -f "$path" ]; then
        ls -lh "$path" 2>/dev/null | awk '{print $5}' || echo "0B"
    else
        echo "0B"
    fi
}

# 计算目录中的文件数量
count_files() {
    local path="$1"
    if [ -d "$path" ]; then
        find "$path" -type f 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# 确认操作
confirm_action() {
    local message="$1"
    if [ "$FORCE_CLEAN" = true ]; then
        return 0
    fi
    
    echo -e "${YELLOW}$message${NC}"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# 清理日志文件
clean_logs() {
    log_info "清理日志文件..."
    
    local logs_dir="logs"
    if [ ! -d "$logs_dir" ]; then
        log_warning "日志目录不存在: $logs_dir"
        return
    fi
    
    local total_size=$(get_size "$logs_dir")
    local file_count=$(count_files "$logs_dir")
    
    log_info "日志目录大小: $total_size，文件数量: $file_count"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[预览] 将要清理的日志文件:"
        find "$logs_dir" -name "*.log" -mtime +$KEEP_DAYS -type f 2>/dev/null | head -10
        local old_count=$(find "$logs_dir" -name "*.log" -mtime +$KEEP_DAYS -type f 2>/dev/null | wc -l | tr -d ' ')
        log_info "[预览] 将删除 $old_count 个日志文件"
        return
    fi
    
    if confirm_action "将清理 ${KEEP_DAYS} 天前的日志文件"; then
        # 清理旧日志文件
        find "$logs_dir" -name "*.log" -mtime +$KEEP_DAYS -type f -delete 2>/dev/null || true
        
        # 清理空的日志文件
        find "$logs_dir" -name "*.log" -size 0 -type f -delete 2>/dev/null || true
        
        # 压缩大日志文件
        find "$logs_dir" -name "*.log" -size +10M -type f -exec gzip {} \; 2>/dev/null || true
        
        local new_size=$(get_size "$logs_dir")
        local new_count=$(count_files "$logs_dir")
        
        log_success "日志清理完成"
        log_info "清理后大小: $new_size，文件数量: $new_count"
    fi
}

# 清理缓存文件
clean_cache() {
    log_info "清理缓存文件..."
    
    local cache_dirs=("data_cache" "__pycache__" ".pytest_cache" "src/__pycache__")
    local total_cleaned=0
    
    for cache_dir in "${cache_dirs[@]}"; do
        if [ -d "$cache_dir" ]; then
            local size=$(get_size "$cache_dir")
            local count=$(count_files "$cache_dir")
            
            log_info "缓存目录: $cache_dir，大小: $size，文件数量: $count"
            
            if [ "$DRY_RUN" = true ]; then
                log_info "[预览] 将清理缓存目录: $cache_dir"
                continue
            fi
            
            if confirm_action "将清理缓存目录: $cache_dir"; then
                rm -rf "$cache_dir"/* 2>/dev/null || true
                log_success "已清理缓存目录: $cache_dir"
                total_cleaned=$((total_cleaned + 1))
            fi
        fi
    done
    
    # 清理Python字节码文件
    if [ "$DRY_RUN" = true ]; then
        local pyc_count=$(find . -name "*.pyc" -type f 2>/dev/null | wc -l | tr -d ' ')
        log_info "[预览] 将删除 $pyc_count 个 .pyc 文件"
    else
        find . -name "*.pyc" -type f -delete 2>/dev/null || true
        find . -name "*.pyo" -type f -delete 2>/dev/null || true
        log_success "已清理Python字节码文件"
    fi
    
    if [ "$DRY_RUN" = false ]; then
        log_success "缓存清理完成，共清理 $total_cleaned 个目录"
    fi
}

# 清理临时文件
clean_temp() {
    log_info "清理临时文件..."
    
    local temp_patterns=("*.tmp" "*.temp" "*.bak" "*.swp" "*.swo" "*~" ".DS_Store")
    local total_deleted=0
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[预览] 将要清理的临时文件:"
        for pattern in "${temp_patterns[@]}"; do
            find . -name "$pattern" -type f 2>/dev/null | head -5
        done
        return
    fi
    
    if confirm_action "将清理临时文件"; then
        for pattern in "${temp_patterns[@]}"; do
            local deleted=$(find . -name "$pattern" -type f -delete -print 2>/dev/null | wc -l | tr -d ' ')
            total_deleted=$((total_deleted + deleted))
        done
        
        # 清理空目录
        find . -type d -empty -delete 2>/dev/null || true
        
        log_success "临时文件清理完成，共删除 $total_deleted 个文件"
    fi
}

# 清理旧备份文件
clean_backups() {
    log_info "清理旧备份文件..."
    
    local backup_dir="backups"
    if [ ! -d "$backup_dir" ]; then
        log_warning "备份目录不存在: $backup_dir"
        return
    fi
    
    local total_size=$(get_size "$backup_dir")
    local file_count=$(count_files "$backup_dir")
    
    log_info "备份目录大小: $total_size，文件数量: $file_count"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[预览] 将要清理的备份文件:"
        find "$backup_dir" -name "*.tar.gz" -mtime +$KEEP_DAYS -type f 2>/dev/null | head -10
        local old_count=$(find "$backup_dir" -name "*.tar.gz" -mtime +$KEEP_DAYS -type f 2>/dev/null | wc -l | tr -d ' ')
        log_info "[预览] 将删除 $old_count 个备份文件"
        return
    fi
    
    if confirm_action "将清理 ${KEEP_DAYS} 天前的备份文件"; then
        find "$backup_dir" -name "*.tar.gz" -mtime +$KEEP_DAYS -type f -delete 2>/dev/null || true
        find "$backup_dir" -name "*.zip" -mtime +$KEEP_DAYS -type f -delete 2>/dev/null || true
        
        local new_size=$(get_size "$backup_dir")
        local new_count=$(count_files "$backup_dir")
        
        log_success "备份清理完成"
        log_info "清理后大小: $new_size，文件数量: $new_count"
    fi
}

# 清理Docker资源
clean_docker() {
    log_info "清理Docker资源..."
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker 未安装，跳过Docker清理"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[预览] Docker资源使用情况:"
        docker system df 2>/dev/null || true
        return
    fi
    
    if confirm_action "将清理Docker资源（悬空镜像、容器、网络、卷）"; then
        # 清理停止的容器
        docker container prune -f 2>/dev/null || true
        
        # 清理悬空镜像
        docker image prune -f 2>/dev/null || true
        
        # 清理未使用的网络
        docker network prune -f 2>/dev/null || true
        
        # 清理未使用的卷
        docker volume prune -f 2>/dev/null || true
        
        # 清理构建缓存
        docker builder prune -f 2>/dev/null || true
        
        log_success "Docker资源清理完成"
        
        log_info "清理后Docker资源使用情况:"
        docker system df 2>/dev/null || true
    fi
}

# 清理Python缓存
clean_pycache() {
    log_info "清理Python缓存..."
    
    if [ "$DRY_RUN" = true ]; then
        local pycache_dirs=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l | tr -d ' ')
        local pyc_files=$(find . -name "*.pyc" -type f 2>/dev/null | wc -l | tr -d ' ')
        log_info "[预览] 将删除 $pycache_dirs 个 __pycache__ 目录和 $pyc_files 个 .pyc 文件"
        return
    fi
    
    if confirm_action "将清理Python缓存文件"; then
        # 清理 __pycache__ 目录
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        
        # 清理 .pyc 文件
        find . -name "*.pyc" -type f -delete 2>/dev/null || true
        find . -name "*.pyo" -type f -delete 2>/dev/null || true
        
        # 清理 .pytest_cache
        find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
        
        log_success "Python缓存清理完成"
    fi
}

# 显示清理摘要
show_summary() {
    log_info "生成清理摘要..."
    
    local summary_file="logs/clean_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "=== Firstrade交易系统清理摘要 ==="
        echo "清理时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "保留天数: $KEEP_DAYS"
        echo ""
        
        echo "目录大小统计:"
        [ -d "logs" ] && echo "  日志目录: $(get_size logs)"
        [ -d "data_cache" ] && echo "  缓存目录: $(get_size data_cache)"
        [ -d "backups" ] && echo "  备份目录: $(get_size backups)"
        echo ""
        
        echo "磁盘使用情况:"
        df -h .
        echo ""
        
        echo "清理操作:"
        [ "$CLEAN_LOGS" = true ] && echo "  ✓ 日志文件"
        [ "$CLEAN_CACHE" = true ] && echo "  ✓ 缓存文件"
        [ "$CLEAN_TEMP" = true ] && echo "  ✓ 临时文件"
        [ "$CLEAN_BACKUPS" = true ] && echo "  ✓ 备份文件"
        [ "$CLEAN_DOCKER" = true ] && echo "  ✓ Docker资源"
        [ "$CLEAN_PYCACHE" = true ] && echo "  ✓ Python缓存"
        
    } > "$summary_file"
    
    log_success "清理摘要已保存: $summary_file"
}

# 主函数
main() {
    # 默认参数
    CLEAN_ALL=false
    CLEAN_LOGS=false
    CLEAN_CACHE=false
    CLEAN_TEMP=false
    CLEAN_BACKUPS=false
    CLEAN_DOCKER=false
    CLEAN_PYCACHE=false
    DRY_RUN=false
    FORCE_CLEAN=false
    KEEP_DAYS=7
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                CLEAN_ALL=true
                shift
                ;;
            -l|--logs)
                CLEAN_LOGS=true
                shift
                ;;
            -c|--cache)
                CLEAN_CACHE=true
                shift
                ;;
            -t|--temp)
                CLEAN_TEMP=true
                shift
                ;;
            -b|--backups)
                CLEAN_BACKUPS=true
                shift
                ;;
            -d|--docker)
                CLEAN_DOCKER=true
                shift
                ;;
            -p|--pycache)
                CLEAN_PYCACHE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --keep-days)
                KEEP_DAYS="$2"
                shift 2
                ;;
            --force)
                FORCE_CLEAN=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果选择清理所有内容
    if [ "$CLEAN_ALL" = true ]; then
        CLEAN_LOGS=true
        CLEAN_CACHE=true
        CLEAN_TEMP=true
        CLEAN_BACKUPS=true
        CLEAN_DOCKER=true
        CLEAN_PYCACHE=true
    fi
    
    # 如果没有指定任何清理选项，默认清理基本内容
    if [ "$CLEAN_LOGS" = false ] && [ "$CLEAN_CACHE" = false ] && [ "$CLEAN_TEMP" = false ] && \
       [ "$CLEAN_BACKUPS" = false ] && [ "$CLEAN_DOCKER" = false ] && [ "$CLEAN_PYCACHE" = false ]; then
        CLEAN_LOGS=true
        CLEAN_CACHE=true
        CLEAN_TEMP=true
    fi
    
    log_info "开始清理Firstrade交易系统..."
    
    if [ "$DRY_RUN" = true ]; then
        log_warning "预览模式 - 不会实际删除文件"
    fi
    
    # 创建日志目录
    mkdir -p logs
    
    # 执行清理操作
    [ "$CLEAN_LOGS" = true ] && clean_logs
    [ "$CLEAN_CACHE" = true ] && clean_cache
    [ "$CLEAN_TEMP" = true ] && clean_temp
    [ "$CLEAN_BACKUPS" = true ] && clean_backups
    [ "$CLEAN_DOCKER" = true ] && clean_docker
    [ "$CLEAN_PYCACHE" = true ] && clean_pycache
    
    # 生成清理摘要
    if [ "$DRY_RUN" = false ]; then
        show_summary
    fi
    
    log_success "清理操作完成"
}

# 执行主函数
main "$@"