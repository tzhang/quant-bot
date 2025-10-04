# 量化交易系统 Makefile
# 提供常用的开发、测试、部署命令

.PHONY: help install install-dev clean test test-fast test-cov lint format type-check security-check pre-commit setup-dev run-api run-streamlit docker-build docker-run backup restore db-setup

# 默认目标
help:
	@echo "量化交易系统开发工具"
	@echo ""
	@echo "可用命令:"
	@echo "  install        - 安装生产依赖"
	@echo "  install-dev    - 安装开发依赖"
	@echo "  clean          - 清理临时文件"
	@echo "  test           - 运行所有测试"
	@echo "  test-fast      - 运行快速测试"
	@echo "  test-cov       - 运行测试并生成覆盖率报告"
	@echo "  lint           - 代码检查"
	@echo "  format         - 代码格式化"
	@echo "  type-check     - 类型检查"
	@echo "  security-check - 安全检查"
	@echo "  pre-commit     - 运行预提交检查"
	@echo "  setup-dev      - 设置开发环境"
	@echo "  run-api        - 启动API服务"
	@echo "  run-streamlit  - 启动Streamlit应用"
	@echo "  docker-build   - 构建Docker镜像"
	@echo "  docker-run     - 运行Docker容器"
	@echo "  backup         - 备份数据"
	@echo "  restore        - 恢复数据"
	@echo "  db-setup       - 统一创建数据库、建表并验证"
	@echo "  data-fetch-nasdaq - 批量抓取NASDAQ-100数据并入库"

# 变量定义
PYTHON := python3
PIP := pip3
VENV := venv
SOURCE_DIR := src
TEST_DIR := tests
COVERAGE_DIR := htmlcov

# 检查虚拟环境
check-venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "虚拟环境不存在，正在创建..."; \
		$(PYTHON) -m venv $(VENV); \
	fi

# 激活虚拟环境的命令
ACTIVATE := . $(VENV)/bin/activate

# 安装生产依赖
install: check-venv
	$(ACTIVATE) && $(PIP) install --upgrade pip
	$(ACTIVATE) && $(PIP) install -r requirements.txt

# 安装开发依赖
install-dev: install
	$(ACTIVATE) && $(PIP) install -e ".[dev,test,docs]"
	$(ACTIVATE) && pre-commit install

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf $(COVERAGE_DIR)/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	@echo "清理完成"

# 运行所有测试
test: check-venv
	$(ACTIVATE) && pytest $(TEST_DIR) -v

# 运行快速测试（排除慢速测试）
test-fast: check-venv
	$(ACTIVATE) && pytest $(TEST_DIR) -v -m "not slow"

# 运行测试并生成覆盖率报告
test-cov: check-venv
	$(ACTIVATE) && pytest $(TEST_DIR) --cov=$(SOURCE_DIR) --cov-report=html --cov-report=term-missing
	@echo "覆盖率报告已生成: $(COVERAGE_DIR)/index.html"

# 代码检查
lint: check-venv
	@echo "运行代码检查..."
	$(ACTIVATE) && flake8 $(SOURCE_DIR) $(TEST_DIR)
	$(ACTIVATE) && isort --check-only $(SOURCE_DIR) $(TEST_DIR)
	$(ACTIVATE) && black --check $(SOURCE_DIR) $(TEST_DIR)

# 代码格式化
format: check-venv
	@echo "格式化代码..."
	$(ACTIVATE) && isort $(SOURCE_DIR) $(TEST_DIR)
	$(ACTIVATE) && black $(SOURCE_DIR) $(TEST_DIR)
	@echo "代码格式化完成"

# 类型检查
type-check: check-venv
	@echo "运行类型检查..."
	$(ACTIVATE) && mypy $(SOURCE_DIR)

# 安全检查
security-check: check-venv
	@echo "运行安全检查..."
	$(ACTIVATE) && bandit -r $(SOURCE_DIR)

# 运行所有预提交检查
pre-commit: format lint type-check security-check test-fast
	@echo "所有检查通过！"

# 设置开发环境
setup-dev: install-dev
	@echo "设置开发环境..."
	@if [ ! -f ".env" ]; then \
		cp .env.example .env; \
		echo "已创建.env文件，请根据需要修改配置"; \
	fi
	mkdir -p logs
	mkdir -p data
	mkdir -p cache
	mkdir -p backtest_results
	mkdir -p performance_reports
	@echo "开发环境设置完成"

# 启动API服务
run-api: check-venv
	@echo "启动API服务..."
	$(ACTIVATE) && cd $(SOURCE_DIR) && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 启动Streamlit应用
run-streamlit: check-venv
	@echo "启动Streamlit应用..."
	$(ACTIVATE) && streamlit run src/web/streamlit_app.py

# 数据库相关命令
db-init: check-venv
	@echo "初始化数据库..."
	$(ACTIVATE) && python src/database/init_db.py

db-migrate: check-venv
	@echo "运行数据库迁移..."
	$(ACTIVATE) && alembic upgrade head

db-reset: check-venv
	@echo "重置数据库..."
	@PG_BIN="$$(brew --prefix postgresql@17)/bin"; \
	set -a; [ -f ".env" ] && . .env; set +a; \
	echo "启动PostgreSQL服务..."; \
	brew services start postgresql@17 >/dev/null 2>&1 || true; \
	echo "执行删除并重建所有表..."; \
	$(ACTIVATE) && python src/database/reset_db.py; \
	echo "验证表是否存在..."; \
	"$$PG_BIN"/psql -h "$${POSTGRES_HOST:-localhost}" -p "$${POSTGRES_PORT:-5432}" -U "$${POSTGRES_USER:-$$USER}" -d "$${POSTGRES_DB:-quant_trading}" -c "\\dt"

# 统一创建数据库、建表并验证
db-setup: check-venv
	@echo "统一创建数据库、建表并验证..."
	@PG_BIN="$$(brew --prefix postgresql@17)/bin"; \
	set -a; [ -f ".env" ] && . .env; set +a; \
	echo "启动PostgreSQL服务..."; \
	brew services start postgresql@17 >/dev/null 2>&1 || true; \
	echo "创建数据库 '$$POSTGRES_DB'（若已存在则忽略）..."; \
	"$$PG_BIN"/createdb -h "$${POSTGRES_HOST:-localhost}" -p "$${POSTGRES_PORT:-5432}" -U "$${POSTGRES_USER:-$$USER}" "$${POSTGRES_DB:-quant_trading}" >/dev/null 2>&1 || echo "数据库已存在或无法创建，继续"; \
	echo "运行初始化脚本建表..."; \
	$(ACTIVATE) && python src/database/init_db.py; \
	echo "验证表是否存在..."; \
	"$$PG_BIN"/psql -h "$${POSTGRES_HOST:-localhost}" -p "$${POSTGRES_PORT:-5432}" -U "$${POSTGRES_USER:-$$USER}" -d "$${POSTGRES_DB:-quant_trading}" -c "\\dt"

# 数据库备份（支持日志与压缩，自定义格式 .dump/.dump.gz）
db-backup: check-venv
	@echo "备份数据库..."
	@PG_BIN="$$(brew --prefix postgresql@17)/bin"; \
	CLI_BACKUP_DIR="$${BACKUP_DIR}"; CLI_BACKUP_COMPRESS="$${BACKUP_COMPRESS}"; \
	set -a; [ -f ".env" ] && . .env; set +a; \
	[ -n "$${CLI_BACKUP_DIR}" ] && BACKUP_DIR="$${CLI_BACKUP_DIR}"; \
	[ -n "$${CLI_BACKUP_COMPRESS}" ] && BACKUP_COMPRESS="$${CLI_BACKUP_COMPRESS}"; \
	mkdir -p "$${BACKUP_DIR:-backups}"; mkdir -p logs; \
	DB_NAME="$${POSTGRES_DB:-quant_trading}"; STAMP="$$(date +%Y%m%d_%H%M%S)"; \
	LOG_FILE="logs/db_backup_$$STAMP.log"; \
	echo "日志文件: $$LOG_FILE" | tee -a "$$LOG_FILE"; \
	echo "数据库: $$DB_NAME, 目录: $${BACKUP_DIR:-backups}, 压缩: $${BACKUP_COMPRESS:-false}" | tee -a "$$LOG_FILE"; \
	if [ "$${BACKUP_COMPRESS:-false}" = "true" ] || [ "$${BACKUP_COMPRESS:-0}" = "1" ]; then \
		BACKUP_FILE="$${BACKUP_DIR:-backups}/$$DB_NAME_$${STAMP}.dump.gz"; \
		echo "备份文件: $$BACKUP_FILE" | tee -a "$$LOG_FILE"; \
		"$$PG_BIN"/pg_dump -h "$${POSTGRES_HOST:-localhost}" -p "$${POSTGRES_PORT:-5432}" -U "$${POSTGRES_USER:-$$USER}" -d "$$DB_NAME" --format=custom --verbose 2>> "$$LOG_FILE" | gzip > "$$BACKUP_FILE"; \
	else \
		BACKUP_FILE="$${BACKUP_DIR:-backups}/$$DB_NAME_$${STAMP}.dump"; \
		echo "备份文件: $$BACKUP_FILE" | tee -a "$$LOG_FILE"; \
		"$$PG_BIN"/pg_dump -h "$${POSTGRES_HOST:-localhost}" -p "$${POSTGRES_PORT:-5432}" -U "$${POSTGRES_USER:-$$USER}" -d "$$DB_NAME" --format=custom --verbose --file="$$BACKUP_FILE" 2>&1 | tee -a "$$LOG_FILE"; \
	fi; \
	echo "✅ 备份完成" | tee -a "$$LOG_FILE"

# 数据库恢复（支持 .dump/.dump.gz，使用 pg_restore 并记录日志）
db-restore: check-venv
	@echo "从备份文件恢复数据库..."
	@if [ -z "$(file)" ]; then \
		echo "❌ 请提供备份文件: make db-restore file=backups/quant_trading_YYYYMMDD_HHMMSS.dump[.gz]"; \
		exit 1; \
	fi
	@PG_BIN="$$(brew --prefix postgresql@17)/bin"; \
	CLI_BACKUP_DIR="$${BACKUP_DIR}"; \
	set -a; [ -f ".env" ] && . .env; set +a; \
	[ -n "$${CLI_BACKUP_DIR}" ] && BACKUP_DIR="$${CLI_BACKUP_DIR}"; \
	mkdir -p logs; STAMP="$$(date +%Y%m%d_%H%M%S)"; LOG_FILE="logs/db_restore_$$STAMP.log"; \
	echo "日志文件: $$LOG_FILE" | tee -a "$$LOG_FILE"; \
	echo "目标数据库: $${POSTGRES_DB:-quant_trading}" | tee -a "$$LOG_FILE"; \
	if echo "$(file)" | grep -qE "\\.gz$$"; then \
		echo "检测到压缩备份，使用管道解压后恢复" | tee -a "$$LOG_FILE"; \
		gunzip -c "$(file)" | "$$PG_BIN"/pg_restore -h "$${POSTGRES_HOST:-localhost}" -p "$${POSTGRES_PORT:-5432}" -U "$${POSTGRES_USER:-$$USER}" -d "$${POSTGRES_DB:-quant_trading}" --clean --if-exists --no-owner --no-privileges --verbose 2>&1 | tee -a "$$LOG_FILE"; \
	else \
		echo "使用未压缩备份进行恢复" | tee -a "$$LOG_FILE"; \
		"$$PG_BIN"/pg_restore -h "$${POSTGRES_HOST:-localhost}" -p "$${POSTGRES_PORT:-5432}" -U "$${POSTGRES_USER:-$$USER}" -d "$${POSTGRES_DB:-quant_trading}" --clean --if-exists --no-owner --no-privileges --verbose "$(file)" 2>&1 | tee -a "$$LOG_FILE"; \
	fi; \
	echo "✅ 恢复完成" | tee -a "$$LOG_FILE"

# Docker相关命令
docker-build:
	@echo "构建Docker镜像..."
	docker build -t quant-trading-system .

docker-run:
	@echo "运行Docker容器..."
	docker run -p 8000:8000 -p 8501:8501 quant-trading-system

docker-compose-up:
	@echo "启动Docker Compose服务..."
	docker-compose up -d

docker-compose-down:
	@echo "停止Docker Compose服务..."
	docker-compose down

# 数据备份和恢复
backup:
	@echo "备份数据..."
	mkdir -p backups
	$(ACTIVATE) && python scripts/backup_data.py

restore:
	@echo "恢复数据..."
	$(ACTIVATE) && python scripts/restore_data.py

# 性能测试
perf-test: check-venv
	@echo "运行性能测试..."
	$(ACTIVATE) && pytest $(TEST_DIR) -v -m "performance"

# 批量抓取NASDAQ-100数据并入库（支持环境变量覆盖）
data-fetch-nasdaq: check-venv
	@echo "批量抓取NASDAQ-100数据并入库..."
	@set -a; [ -f ".env" ] && . .env; set +a; \
	$(ACTIVATE) && MAX_TICKERS="$${MAX_TICKERS:-100}" BATCH_SIZE="$${BATCH_SIZE:-1000}" THROTTLE_SEC="$${THROTTLE_SEC:-0.5}" USE_PARQUET="$${USE_PARQUET:-false}" DATA_START_DATE="$${DATA_START_DATE}" DATA_END_DATE="$${DATA_END_DATE}" DATA_CACHE_DIR="$${DATA_CACHE_DIR:-data_cache/nasdaq}" python src/data/fetch_nasdaq.py

# 生成文档
docs: check-venv
	@echo "生成文档..."
	$(ACTIVATE) && cd docs && make html
	@echo "文档已生成: docs/_build/html/index.html"

# 发布准备
release-check: clean lint type-check security-check test-cov
	@echo "发布前检查完成"

# 构建分发包
build: clean
	@echo "构建分发包..."
	$(ACTIVATE) && python -m build

# 上传到PyPI（测试）
upload-test: build
	@echo "上传到PyPI测试环境..."
	$(ACTIVATE) && twine upload --repository testpypi dist/*

# 上传到PyPI（生产）
upload: build
	@echo "上传到PyPI生产环境..."
	$(ACTIVATE) && twine upload dist/*

# 监控日志
logs:
	@echo "查看系统日志..."
	tail -f logs/quant_system.log

# 系统状态检查
status: check-venv
	@echo "检查系统状态..."
	$(ACTIVATE) && python scripts/health_check.py

# 更新依赖
update-deps: check-venv
	@echo "更新依赖包..."
	$(ACTIVATE) && pip list --outdated
	$(ACTIVATE) && pip-review --auto

# 生成需求文件
freeze: check-venv
	@echo "生成当前环境的需求文件..."
	$(ACTIVATE) && pip freeze > requirements-freeze.txt

# 代码统计
stats:
	@echo "代码统计信息:"
	@find $(SOURCE_DIR) -name "*.py" | xargs wc -l | tail -1
	@echo "测试文件:"
	@find $(TEST_DIR) -name "*.py" | xargs wc -l | tail -1

# 快速开始（新开发者）
quickstart: setup-dev
	@echo "快速开始完成！"
	@echo "下一步:"
	@echo "1. 修改 .env 文件中的配置"
	@echo "2. 运行 'make db-init' 初始化数据库"
	@echo "3. 运行 'make test' 确保一切正常"
	@echo "4. 运行 'make run-api' 启动API服务"
	@echo "5. 运行 'make run-streamlit' 启动Web界面"