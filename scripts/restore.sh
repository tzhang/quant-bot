#!/bin/bash

# Firstrade 交易系统数据恢复脚本
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
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RESTORE_DIR="${RESTORE_DIR:-./restore_temp}"

# 数据库配置
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-trading_db}"
DB_USER="${DB_USER:-trader}"
DB_PASSWORD="${DB_PASSWORD:-password}"

# Redis配置
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

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

# 检查依赖
check_dependencies() {
    log_info "检查恢复依赖..."
    
    # 检查pg_restore
    if ! command -v pg_restore &> /dev/null; then
        log_error "pg_restore 未安装，请安装 PostgreSQL 客户端工具"
        exit 1
    fi
    
    # 检查psql
    if ! command -v psql &> /dev/null; then
        log_error "psql 未安装，请安装 PostgreSQL 客户端工具"
        exit 1
    fi
    
    # 检查redis-cli
    if ! command -v redis-cli &> /dev/null; then
        log_error "redis-cli 未安装，请安装 Redis 客户端工具"
        exit 1
    fi
    
    # 检查tar
    if ! command -v tar &> /dev/null; then
        log_error "tar 未安装"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 列出可用备份
list_backups() {
    log_info "可用备份文件:"
    echo ""
    
    if [ ! -d "${BACKUP_DIR}" ]; then
        log_error "备份目录不存在: ${BACKUP_DIR}"
        exit 1
    fi
    
    local backups=$(find "${BACKUP_DIR}" -name "trading_system_backup_*.tar.gz" -type f | sort -r)
    
    if [ -z "${backups}" ]; then
        log_error "未找到备份文件"
        exit 1
    fi
    
    local count=1
    for backup in ${backups}; do
        local filename=$(basename "${backup}")
        local size=$(du -sh "${backup}" | cut -f1)
        local date=$(stat -f%Sm -t"%Y-%m-%d %H:%M:%S" "${backup}" 2>/dev/null || stat -c%y "${backup}" 2>/dev/null | cut -d' ' -f1-2)
        echo "${count}. ${filename} (${size}, ${date})"
        count=$((count + 1))
    done
    echo ""
}

# 验证备份文件
verify_backup_file() {
    local backup_file="$1"
    
    log_info "验证备份文件: ${backup_file}"
    
    # 检查文件是否存在
    if [ ! -f "${backup_file}" ]; then
        log_error "备份文件不存在: ${backup_file}"
        exit 1
    fi
    
    # 检查文件完整性
    if ! tar -tzf "${backup_file}" > /dev/null 2>&1; then
        log_error "备份文件损坏: ${backup_file}"
        exit 1
    fi
    
    log_success "备份文件验证通过"
}

# 解压备份文件
extract_backup() {
    local backup_file="$1"
    
    log_info "解压备份文件..."
    
    # 创建临时恢复目录
    rm -rf "${RESTORE_DIR}"
    mkdir -p "${RESTORE_DIR}"
    
    # 解压备份文件
    tar -xzf "${backup_file}" -C "${RESTORE_DIR}"
    
    # 查找解压后的目录
    local backup_name=$(basename "${backup_file}" .tar.gz)
    EXTRACTED_DIR="${RESTORE_DIR}/${backup_name}"
    
    if [ ! -d "${EXTRACTED_DIR}" ]; then
        log_error "解压后的目录不存在: ${EXTRACTED_DIR}"
        exit 1
    fi
    
    log_success "备份文件解压完成"
}

# 显示备份信息
show_backup_info() {
    local metadata_file="${EXTRACTED_DIR}/backup_metadata.json"
    
    if [ -f "${metadata_file}" ]; then
        log_info "备份信息:"
        echo ""
        
        # 使用python解析JSON（如果可用）
        if command -v python3 &> /dev/null; then
            python3 -c "
import json
import sys

try:
    with open('${metadata_file}', 'r') as f:
        data = json.load(f)
    
    print(f\"📁 备份名称: {data.get('backup_name', 'N/A')}\")
    print(f\"⏰ 备份时间: {data.get('date', 'N/A')}\")
    print(f\"🖥️  主机名: {data.get('system_info', {}).get('hostname', 'N/A')}\")
    print(f\"🔗 Git提交: {data.get('git_commit', 'N/A')}\")
    print(f\"📊 备份大小: {data.get('backup_size', 'N/A')}\")
    print(f\"🗃️  数据库: {data.get('database_info', {}).get('postgres_database', 'N/A')}\")
    
    components = data.get('backup_components', [])
    if components:
        print(f\"📦 备份组件: {', '.join(components)}\")
        
except Exception as e:
    print(f\"无法解析备份元数据: {e}\")
"
        else
            # 简单显示文件内容
            cat "${metadata_file}"
        fi
        echo ""
    else
        log_warning "未找到备份元数据文件"
    fi
}

# 确认恢复操作
confirm_restore() {
    log_warning "⚠️  恢复操作将覆盖现有数据！"
    echo ""
    echo "即将恢复的数据:"
    echo "- PostgreSQL数据库: ${DB_NAME}"
    echo "- Redis数据"
    echo "- 应用数据文件"
    echo "- Docker数据卷"
    echo ""
    
    read -p "确认继续恢复操作？(yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "恢复操作已取消"
        exit 0
    fi
}

# 停止相关服务
stop_services() {
    log_info "停止相关服务..."
    
    # 停止Docker Compose服务
    if [ -f "docker-compose.yml" ] && command -v docker-compose &> /dev/null; then
        docker-compose down || log_warning "Docker Compose服务停止失败"
    fi
    
    # 等待服务完全停止
    sleep 5
    
    log_success "服务停止完成"
}

# 恢复PostgreSQL数据库
restore_postgres() {
    log_info "恢复PostgreSQL数据库..."
    
    local postgres_backup="${EXTRACTED_DIR}/postgres_*.sql.custom"
    local postgres_sql="${EXTRACTED_DIR}/postgres_*.sql"
    
    # 设置密码环境变量
    export PGPASSWORD="${DB_PASSWORD}"
    
    # 检查数据库连接
    if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres -c "SELECT 1;" > /dev/null 2>&1; then
        log_error "无法连接到PostgreSQL数据库"
        exit 1
    fi
    
    # 删除现有数据库（如果存在）
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
        -c "DROP DATABASE IF EXISTS ${DB_NAME};" || true
    
    # 创建新数据库
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
        -c "CREATE DATABASE ${DB_NAME};"
    
    # 恢复数据库
    if ls ${postgres_backup} 1> /dev/null 2>&1; then
        # 使用custom格式恢复
        pg_restore -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
            --verbose --clean --if-exists --create \
            ${postgres_backup}
    elif ls ${postgres_sql} 1> /dev/null 2>&1; then
        # 使用SQL格式恢复
        psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
            -f ${postgres_sql}
    else
        log_error "未找到PostgreSQL备份文件"
        exit 1
    fi
    
    # 清除密码环境变量
    unset PGPASSWORD
    
    log_success "PostgreSQL数据库恢复完成"
}

# 恢复Redis数据
restore_redis() {
    log_info "恢复Redis数据..."
    
    local redis_backup="${EXTRACTED_DIR}/redis_*.rdb"
    
    if ls ${redis_backup} 1> /dev/null 2>&1; then
        # 停止Redis服务
        if command -v docker &> /dev/null && docker ps | grep -q redis; then
            # Docker中的Redis
            docker stop $(docker ps --filter "name=redis" --format "{{.Names}}") || true
            
            # 复制备份文件到Redis数据目录
            docker cp ${redis_backup} $(docker ps -a --filter "name=redis" --format "{{.Names}}"):/data/dump.rdb
            
            # 重启Redis
            docker start $(docker ps -a --filter "name=redis" --format "{{.Names}}")
        else
            # 本地Redis
            local redis_dir=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" CONFIG GET dir | tail -1)
            
            # 停止Redis
            redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" SHUTDOWN NOSAVE || true
            
            # 复制备份文件
            cp ${redis_backup} "${redis_dir}/dump.rdb"
            
            # 重启Redis（需要根据系统调整）
            if command -v systemctl &> /dev/null; then
                systemctl start redis || true
            elif command -v service &> /dev/null; then
                service redis-server start || true
            else
                log_warning "请手动启动Redis服务"
            fi
        fi
        
        # 等待Redis启动
        sleep 5
        
        # 验证Redis连接
        if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping > /dev/null 2>&1; then
            log_success "Redis数据恢复完成"
        else
            log_error "Redis恢复后无法连接"
        fi
    else
        log_warning "未找到Redis备份文件"
    fi
}

# 恢复应用数据
restore_app_data() {
    log_info "恢复应用数据..."
    
    local app_data_dir="${EXTRACTED_DIR}/app_data"
    
    if [ -d "${app_data_dir}" ]; then
        # 备份现有数据
        local backup_timestamp=$(date +"%Y%m%d_%H%M%S")
        
        # 恢复配置文件
        if [ -d "${app_data_dir}/config" ]; then
            if [ -d "./config" ]; then
                mv ./config "./config.backup.${backup_timestamp}"
            fi
            cp -r "${app_data_dir}/config" ./
            log_info "配置文件恢复完成"
        fi
        
        # 恢复数据缓存
        if [ -d "${app_data_dir}/data_cache" ]; then
            if [ -d "./data_cache" ]; then
                mv ./data_cache "./data_cache.backup.${backup_timestamp}"
            fi
            cp -r "${app_data_dir}/data_cache" ./
            log_info "数据缓存恢复完成"
        fi
        
        # 恢复策略测试结果
        if [ -d "${app_data_dir}/strategy_test_results" ]; then
            if [ -d "./strategy_test_results" ]; then
                mv ./strategy_test_results "./strategy_test_results.backup.${backup_timestamp}"
            fi
            cp -r "${app_data_dir}/strategy_test_results" ./
            log_info "策略测试结果恢复完成"
        fi
        
        # 恢复数据质量报告
        if [ -d "${app_data_dir}/data_quality_reports" ]; then
            if [ -d "./data_quality_reports" ]; then
                mv ./data_quality_reports "./data_quality_reports.backup.${backup_timestamp}"
            fi
            cp -r "${app_data_dir}/data_quality_reports" ./
            log_info "数据质量报告恢复完成"
        fi
        
        # 恢复结果数据
        if [ -d "${app_data_dir}/results" ]; then
            if [ -d "./results" ]; then
                mv ./results "./results.backup.${backup_timestamp}"
            fi
            cp -r "${app_data_dir}/results" ./
            log_info "结果数据恢复完成"
        fi
        
        log_success "应用数据恢复完成"
    else
        log_warning "未找到应用数据备份"
    fi
}

# 恢复Docker数据卷
restore_docker_volumes() {
    log_info "恢复Docker数据卷..."
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker未安装，跳过数据卷恢复"
        return
    fi
    
    local volumes_dir="${EXTRACTED_DIR}/docker_volumes"
    
    if [ -d "${volumes_dir}" ]; then
        for volume_backup in "${volumes_dir}"/*.tar.gz; do
            if [ -f "${volume_backup}" ]; then
                local volume_name=$(basename "${volume_backup}" | sed 's/_[0-9]*_[0-9]*.tar.gz$//')
                
                log_info "恢复数据卷: ${volume_name}"
                
                # 创建数据卷（如果不存在）
                docker volume create "${volume_name}" || true
                
                # 恢复数据卷内容
                docker run --rm -v "${volume_name}:/data" -v "${PWD}/${volume_backup}:/backup.tar.gz" \
                    alpine sh -c "cd /data && tar -xzf /backup.tar.gz --strip-components=1"
            fi
        done
        
        log_success "Docker数据卷恢复完成"
    else
        log_warning "未找到Docker数据卷备份"
    fi
}

# 启动服务
start_services() {
    log_info "启动服务..."
    
    # 启动Docker Compose服务
    if [ -f "docker-compose.yml" ] && command -v docker-compose &> /dev/null; then
        docker-compose up -d
        
        # 等待服务启动
        sleep 10
        
        log_success "服务启动完成"
    else
        log_warning "未找到docker-compose.yml，请手动启动服务"
    fi
}

# 验证恢复结果
verify_restore() {
    log_info "验证恢复结果..."
    
    # 验证PostgreSQL
    export PGPASSWORD="${DB_PASSWORD}"
    if psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT COUNT(*) FROM information_schema.tables;" > /dev/null 2>&1; then
        local table_count=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
        log_success "PostgreSQL验证通过，表数量: ${table_count}"
    else
        log_error "PostgreSQL验证失败"
    fi
    unset PGPASSWORD
    
    # 验证Redis
    if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping > /dev/null 2>&1; then
        local key_count=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" DBSIZE)
        log_success "Redis验证通过，键数量: ${key_count}"
    else
        log_error "Redis验证失败"
    fi
    
    # 验证应用文件
    local file_count=0
    for dir in config data_cache strategy_test_results data_quality_reports results; do
        if [ -d "./${dir}" ]; then
            file_count=$((file_count + 1))
        fi
    done
    log_success "应用数据验证通过，目录数量: ${file_count}"
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."
    
    if [ -d "${RESTORE_DIR}" ]; then
        rm -rf "${RESTORE_DIR}"
        log_success "临时文件清理完成"
    fi
}

# 显示恢复信息
show_restore_info() {
    log_success "恢复完成！"
    echo ""
    echo "=== 恢复信息 ==="
    echo "📁 备份文件: ${BACKUP_FILE}"
    echo "⏰ 恢复时间: $(date)"
    echo "🖥️  主机名: $(hostname)"
    echo ""
    echo "=== 恢复内容 ==="
    echo "✅ PostgreSQL数据库"
    echo "✅ Redis数据"
    echo "✅ 应用数据"
    echo "✅ Docker数据卷"
    echo ""
    echo "=== 服务访问 ==="
    echo "🌐 监控面板: http://localhost:8080"
    echo "📊 Grafana: http://localhost:3000"
    echo "📈 Prometheus: http://localhost:9090"
    echo ""
}

# 主函数
main() {
    log_info "开始恢复 Firstrade 交易系统..."
    
    # 解析命令行参数
    BACKUP_NAME=""
    FORCE_RESTORE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            --restore-dir)
                RESTORE_DIR="$2"
                shift 2
                ;;
            --db-host)
                DB_HOST="$2"
                shift 2
                ;;
            --db-port)
                DB_PORT="$2"
                shift 2
                ;;
            --db-name)
                DB_NAME="$2"
                shift 2
                ;;
            --db-user)
                DB_USER="$2"
                shift 2
                ;;
            --db-password)
                DB_PASSWORD="$2"
                shift 2
                ;;
            --redis-host)
                REDIS_HOST="$2"
                shift 2
                ;;
            --redis-port)
                REDIS_PORT="$2"
                shift 2
                ;;
            --force)
                FORCE_RESTORE=true
                shift
                ;;
            --help|-h)
                echo "用法: $0 [备份名称] [选项]"
                echo ""
                echo "参数:"
                echo "  备份名称                  要恢复的备份文件名（不含扩展名）"
                echo ""
                echo "选项:"
                echo "  --backup-dir DIR          备份目录 (默认: ./backups)"
                echo "  --restore-dir DIR         临时恢复目录 (默认: ./restore_temp)"
                echo "  --db-host HOST           数据库主机 (默认: localhost)"
                echo "  --db-port PORT           数据库端口 (默认: 5432)"
                echo "  --db-name NAME           数据库名称 (默认: trading_db)"
                echo "  --db-user USER           数据库用户 (默认: trader)"
                echo "  --db-password PASS       数据库密码 (默认: password)"
                echo "  --redis-host HOST        Redis主机 (默认: localhost)"
                echo "  --redis-port PORT        Redis端口 (默认: 6379)"
                echo "  --force                  强制恢复，不显示确认提示"
                echo "  --help, -h               显示帮助信息"
                echo ""
                echo "示例:"
                echo "  $0                                    # 列出可用备份"
                echo "  $0 trading_system_backup_20231201_120000  # 恢复指定备份"
                exit 0
                ;;
            *)
                if [ -z "${BACKUP_NAME}" ]; then
                    BACKUP_NAME="$1"
                else
                    log_error "未知参数: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 检查依赖
    check_dependencies
    
    # 如果没有指定备份名称，列出可用备份
    if [ -z "${BACKUP_NAME}" ]; then
        list_backups
        exit 0
    fi
    
    # 构建备份文件路径
    BACKUP_FILE="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    
    # 验证备份文件
    verify_backup_file "${BACKUP_FILE}"
    
    # 解压备份文件
    extract_backup "${BACKUP_FILE}"
    
    # 显示备份信息
    show_backup_info
    
    # 确认恢复操作
    if [ "${FORCE_RESTORE}" != true ]; then
        confirm_restore
    fi
    
    # 执行恢复流程
    stop_services
    restore_postgres
    restore_redis
    restore_app_data
    restore_docker_volumes
    start_services
    verify_restore
    cleanup
    show_restore_info
}

# 执行主函数
main "$@"