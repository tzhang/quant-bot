# 量化交易系统 Makefile
# 提供常用的开发、测试、部署命令

.PHONY: help install install-dev clean test test-fast test-cov lint format type-check security-check pre-commit setup-dev run-api run-streamlit docker-build docker-run backup restore

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
	$(ACTIVATE) && python -c "from src.data.database import init_db; init_db()"

db-migrate: check-venv
	@echo "运行数据库迁移..."
	$(ACTIVATE) && alembic upgrade head

db-reset: check-venv
	@echo "重置数据库..."
	$(ACTIVATE) && python -c "from src.data.database import reset_db; reset_db()"

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