#!/bin/bash

# Firstrade 交易系统数据备份脚本
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
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="trading_system_backup_${TIMESTAMP}"

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
    log_info "检查备份依赖..."
    
    # 检查pg_dump
    if ! command -v pg_dump &> /dev/null; then
        log_error "pg_dump 未安装，请安装 PostgreSQL 客户端工具"
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

# 创建备份目录
create_backup_dir() {
    log_info "创建备份目录..."
    
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
    
    log_success "备份目录创建完成: ${BACKUP_DIR}/${BACKUP_NAME}"
}

# 备份PostgreSQL数据库
backup_postgres() {
    log_info "备份PostgreSQL数据库..."
    
    local backup_file="${BACKUP_DIR}/${BACKUP_NAME}/postgres_${TIMESTAMP}.sql"
    
    # 设置密码环境变量
    export PGPASSWORD="${DB_PASSWORD}"
    
    # 执行数据库备份
    pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
        --verbose --clean --if-exists --create \
        --format=custom --compress=9 \
        --file="${backup_file}.custom"
    
    # 同时创建SQL格式备份
    pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
        --verbose --clean --if-exists --create \
        --format=plain \
        --file="${backup_file}"
    
    # 清除密码环境变量
    unset PGPASSWORD
    
    log_success "PostgreSQL数据库备份完成"
}

# 备份Redis数据
backup_redis() {
    log_info "备份Redis数据..."
    
    local backup_file="${BACKUP_DIR}/${BACKUP_NAME}/redis_${TIMESTAMP}.rdb"
    
    # 触发Redis保存
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" BGSAVE
    
    # 等待保存完成
    while [ "$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" LASTSAVE)" = "$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" LASTSAVE)" ]; do
        sleep 1
    done
    
    # 复制RDB文件
    if command -v docker &> /dev/null && docker ps | grep -q redis; then
        # 如果Redis运行在Docker中
        docker cp $(docker ps --filter "name=redis" --format "{{.Names}}"):/data/dump.rdb "${backup_file}"
    else
        # 如果Redis运行在本地
        local redis_dir=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" CONFIG GET dir | tail -1)
        cp "${redis_dir}/dump.rdb" "${backup_file}"
    fi
    
    log_success "Redis数据备份完成"
}

# 备份应用数据
backup_app_data() {
    log_info "备份应用数据..."
    
    local app_backup_dir="${BACKUP_DIR}/${BACKUP_NAME}/app_data"
    mkdir -p "${app_backup_dir}"
    
    # 备份配置文件
    if [ -d "./config" ]; then
        cp -r ./config "${app_backup_dir}/"
        log_info "配置文件备份完成"
    fi
    
    # 备份日志文件
    if [ -d "./logs" ]; then
        cp -r ./logs "${app_backup_dir}/"
        log_info "日志文件备份完成"
    fi
    
    # 备份数据缓存
    if [ -d "./data_cache" ]; then
        cp -r ./data_cache "${app_backup_dir}/"
        log_info "数据缓存备份完成"
    fi
    
    # 备份策略测试结果
    if [ -d "./strategy_test_results" ]; then
        cp -r ./strategy_test_results "${app_backup_dir}/"
        log_info "策略测试结果备份完成"
    fi
    
    # 备份数据质量报告
    if [ -d "./data_quality_reports" ]; then
        cp -r ./data_quality_reports "${app_backup_dir}/"
        log_info "数据质量报告备份完成"
    fi
    
    # 备份结果数据
    if [ -d "./results" ]; then
        cp -r ./results "${app_backup_dir}/"
        log_info "结果数据备份完成"
    fi
    
    log_success "应用数据备份完成"
}

# 备份Docker数据卷
backup_docker_volumes() {
    log_info "备份Docker数据卷..."
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker未安装，跳过数据卷备份"
        return
    fi
    
    local volumes_backup_dir="${BACKUP_DIR}/${BACKUP_NAME}/docker_volumes"
    mkdir -p "${volumes_backup_dir}"
    
    # 获取项目相关的数据卷
    local volumes=$(docker volume ls --filter "name=my-quant" --format "{{.Name}}")
    
    if [ -n "${volumes}" ]; then
        for volume in ${volumes}; do
            log_info "备份数据卷: ${volume}"
            docker run --rm -v "${volume}:/data" -v "${PWD}/${volumes_backup_dir}:/backup" \
                alpine tar czf "/backup/${volume}_${TIMESTAMP}.tar.gz" -C /data .
        done
        log_success "Docker数据卷备份完成"
    else
        log_info "未找到相关Docker数据卷"
    fi
}

# 创建备份元数据
create_backup_metadata() {
    log_info "创建备份元数据..."
    
    local metadata_file="${BACKUP_DIR}/${BACKUP_NAME}/backup_metadata.json"
    
    cat > "${metadata_file}" << EOF
{
    "backup_name": "${BACKUP_NAME}",
    "timestamp": "${TIMESTAMP}",
    "date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "version": "1.0.0",
    "system_info": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "user": "$(whoami)"
    },
    "database_info": {
        "postgres_host": "${DB_HOST}",
        "postgres_port": "${DB_PORT}",
        "postgres_database": "${DB_NAME}",
        "redis_host": "${REDIS_HOST}",
        "redis_port": "${REDIS_PORT}"
    },
    "backup_components": [
        "postgres_database",
        "redis_data",
        "application_data",
        "docker_volumes"
    ],
    "backup_size": "$(du -sh ${BACKUP_DIR}/${BACKUP_NAME} | cut -f1)",
    "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
}
EOF
    
    log_success "备份元数据创建完成"
}

# 压缩备份
compress_backup() {
    log_info "压缩备份文件..."
    
    cd "${BACKUP_DIR}"
    tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/"
    
    # 删除原始备份目录
    rm -rf "${BACKUP_NAME}/"
    
    local backup_size=$(du -sh "${BACKUP_NAME}.tar.gz" | cut -f1)
    log_success "备份压缩完成，大小: ${backup_size}"
}

# 清理旧备份
cleanup_old_backups() {
    log_info "清理旧备份文件..."
    
    # 删除超过保留天数的备份文件
    find "${BACKUP_DIR}" -name "trading_system_backup_*.tar.gz" -mtime +${RETENTION_DAYS} -delete
    
    local remaining_backups=$(find "${BACKUP_DIR}" -name "trading_system_backup_*.tar.gz" | wc -l)
    log_success "旧备份清理完成，剩余备份文件: ${remaining_backups} 个"
}

# 验证备份
verify_backup() {
    log_info "验证备份文件..."
    
    local backup_file="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    
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
    
    # 检查文件大小
    local file_size=$(stat -f%z "${backup_file}" 2>/dev/null || stat -c%s "${backup_file}" 2>/dev/null)
    if [ "${file_size}" -lt 1024 ]; then
        log_error "备份文件过小，可能不完整: ${backup_file}"
        exit 1
    fi
    
    log_success "备份文件验证通过"
}

# 发送通知
send_notification() {
    log_info "发送备份通知..."
    
    local backup_file="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    local backup_size=$(du -sh "${backup_file}" | cut -f1)
    
    # 如果配置了邮件通知
    if [ -n "${NOTIFICATION_EMAIL}" ] && command -v mail &> /dev/null; then
        echo "Firstrade交易系统备份完成
        
备份文件: ${backup_file}
备份大小: ${backup_size}
备份时间: $(date)
主机名: $(hostname)

备份包含:
- PostgreSQL数据库
- Redis数据
- 应用数据
- Docker数据卷

请妥善保管备份文件。" | mail -s "交易系统备份完成 - ${TIMESTAMP}" "${NOTIFICATION_EMAIL}"
        
        log_success "邮件通知已发送"
    fi
    
    # 如果配置了Slack通知
    if [ -n "${SLACK_WEBHOOK_URL}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🔄 Firstrade交易系统备份完成\\n📁 文件: ${backup_file}\\n📊 大小: ${backup_size}\\n⏰ 时间: $(date)\"}" \
            "${SLACK_WEBHOOK_URL}"
        
        log_success "Slack通知已发送"
    fi
}

# 显示备份信息
show_backup_info() {
    log_success "备份完成！"
    echo ""
    echo "=== 备份信息 ==="
    echo "📁 备份文件: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    echo "📊 备份大小: $(du -sh ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz | cut -f1)"
    echo "⏰ 备份时间: $(date)"
    echo "🔗 Git提交: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo ""
    echo "=== 备份内容 ==="
    echo "✅ PostgreSQL数据库"
    echo "✅ Redis数据"
    echo "✅ 应用数据"
    echo "✅ Docker数据卷"
    echo ""
    echo "=== 恢复方法 ==="
    echo "解压备份: tar -xzf ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    echo "恢复数据库: ./scripts/restore.sh ${BACKUP_NAME}"
    echo ""
}

# 主函数
main() {
    log_info "开始备份 Firstrade 交易系统..."
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            --retention-days)
                RETENTION_DAYS="$2"
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
            --notification-email)
                NOTIFICATION_EMAIL="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK_URL="$2"
                shift 2
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --backup-dir DIR          备份目录 (默认: ./backups)"
                echo "  --retention-days DAYS     备份保留天数 (默认: 30)"
                echo "  --db-host HOST           数据库主机 (默认: localhost)"
                echo "  --db-port PORT           数据库端口 (默认: 5432)"
                echo "  --db-name NAME           数据库名称 (默认: trading_db)"
                echo "  --db-user USER           数据库用户 (默认: trader)"
                echo "  --db-password PASS       数据库密码 (默认: password)"
                echo "  --redis-host HOST        Redis主机 (默认: localhost)"
                echo "  --redis-port PORT        Redis端口 (默认: 6379)"
                echo "  --notification-email EMAIL  通知邮箱"
                echo "  --slack-webhook URL      Slack Webhook URL"
                echo "  --help, -h               显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 执行备份流程
    check_dependencies
    create_backup_dir
    backup_postgres
    backup_redis
    backup_app_data
    backup_docker_volumes
    create_backup_metadata
    compress_backup
    verify_backup
    cleanup_old_backups
    send_notification
    show_backup_info
}

# 执行主函数
main "$@"