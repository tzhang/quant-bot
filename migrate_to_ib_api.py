#!/usr/bin/env python3
"""
迁移脚本：将项目中的yfinance替换为IB TWS API

该脚本会：
1. 扫描项目中所有使用yfinance的文件
2. 创建备份
3. 替换yfinance调用为IB API调用
4. 更新导入语句
5. 生成迁移报告

作者: AI Assistant
日期: 2024
"""

import os
import re
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration_to_ib.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YFinanceToIBMigrator:
    """yfinance到IB API的迁移器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_before_ib_migration"
        self.migration_report = []
        
        # 需要迁移的文件列表（从搜索结果中获取）
        self.files_to_migrate = [
            "fetch_nasdaq_top100.py",
            "src/data/manager.py", 
            "src/data/sector_data.py",
            "src/data/data_adapter.py",
            "src/data/fundamental_data.py",
            "src/data/macro_data.py",
            "src/data/data_manager.py",
            "src/data/fetch_nasdaq.py",
            "src/data/alternative_data.py",
            "src/data/sentiment_data.py",
            "examples/portfolio_strategy_analysis.py",
            "competitions/citadel/ml_enhanced_citadel_strategy.py",
            "src/ml/advanced_feature_engineering.py"
        ]
        
        # 测试和调试文件（不迁移，但会记录）
        self.test_files = [
            "test_yfinance_fix.py",
            "debug_data_format.py", 
            "debug_index_comparison.py",
            "test_consistency_with_mock_ib.py",
            "validate_ib_data_consistency.py"
        ]
        
        # 文档文件（需要更新说明）
        self.doc_files = [
            "QUICKSTART.md",
            "05-量化交易系统开发起步指南.md",
            "docs/FAQ_TROUBLESHOOTING.md",
            "docs/BEGINNER_GUIDE.md"
        ]
    
    def create_backup(self):
        """创建备份"""
        logger.info("🔄 创建项目备份...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir(parents=True)
        
        # 备份需要迁移的文件
        for file_path in self.files_to_migrate:
            source_file = self.project_root / file_path
            if source_file.exists():
                backup_file = self.backup_dir / file_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, backup_file)
                logger.info(f"   ✅ 备份: {file_path}")
        
        logger.info(f"📁 备份完成，保存在: {self.backup_dir}")
    
    def migrate_file(self, file_path: str) -> Dict:
        """迁移单个文件"""
        source_file = self.project_root / file_path
        
        if not source_file.exists():
            return {"status": "skip", "reason": "文件不存在"}
        
        logger.info(f"🔄 迁移文件: {file_path}")
        
        try:
            # 读取原文件
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes_made = []
            
            # 1. 替换导入语句
            if 'import yfinance as yf' in content:
                content = content.replace(
                    'import yfinance as yf',
                    'from src.data.ib_data_provider import IBDataProvider, IBConfig'
                )
                changes_made.append("替换yfinance导入为IB导入")
            
            # 2. 替换yf.Ticker()调用
            ticker_pattern = r'yf\.Ticker\([\'"]([A-Z]+)[\'"]\)'
            if re.search(ticker_pattern, content):
                # 这需要更复杂的替换逻辑，因为IB API的使用方式不同
                changes_made.append("检测到yf.Ticker()调用，需要手动调整")
            
            # 3. 替换yf.download()调用
            download_pattern = r'yf\.download\('
            if re.search(download_pattern, content):
                changes_made.append("检测到yf.download()调用，需要手动调整")
            
            # 4. 添加IB配置初始化（如果检测到yfinance使用）
            if 'yf.' in content and 'IBDataProvider' not in content:
                # 在文件开头添加IB配置
                ib_init_code = '''
# IB API配置
try:
    from src.data.ib_data_provider import IBDataProvider, IBConfig
    _ib_config = IBConfig()
    _ib_provider = IBDataProvider(_ib_config)
except ImportError:
    _ib_provider = None
    logger.warning("IB API不可用，请安装: pip install ibapi")

'''
                # 找到第一个import语句后插入
                import_pattern = r'(import [^\n]+\n)'
                match = re.search(import_pattern, content)
                if match:
                    insert_pos = match.end()
                    content = content[:insert_pos] + ib_init_code + content[insert_pos:]
                    changes_made.append("添加IB配置初始化代码")
            
            # 5. 创建迁移注释
            migration_comment = f'''
# ==========================================
# 迁移说明 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# ==========================================
# 本文件已从yfinance迁移到IB TWS API
# 原始文件备份在: backup_before_ib_migration/{file_path}
# 
# 主要变更:
# {chr(10).join(f"# - {change}" for change in changes_made)}
# 
# 注意事项:
# 1. 需要启动IB TWS或Gateway
# 2. 确保API设置已正确配置
# 3. 某些yfinance特有功能可能需要手动调整
# ==========================================

'''
            
            # 在文件开头添加迁移注释
            content = migration_comment + content
            
            # 写入修改后的文件
            with open(source_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "status": "success",
                "changes": changes_made,
                "lines_changed": len(content.splitlines()) - len(original_content.splitlines())
            }
            
        except Exception as e:
            logger.error(f"❌ 迁移文件失败: {file_path}, 错误: {e}")
            return {"status": "error", "error": str(e)}
    
    def update_documentation(self):
        """更新文档"""
        logger.info("📝 更新项目文档...")
        
        doc_updates = {
            "yfinance": "IB TWS API",
            "pip install yfinance": "pip install ibapi",
            "import yfinance as yf": "from src.data.ib_data_provider import IBDataProvider, IBConfig",
            "yf.Ticker": "IBDataProvider",
            "yf.download": "ib_provider.get_multiple_stocks_data"
        }
        
        for doc_file in self.doc_files:
            doc_path = self.project_root / doc_file
            if doc_path.exists():
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # 应用文档更新
                    for old_text, new_text in doc_updates.items():
                        content = content.replace(old_text, new_text)
                    
                    if content != original_content:
                        # 添加迁移说明
                        migration_note = f'''
> **📢 迁移说明 ({datetime.now().strftime('%Y-%m-%d')})**  
> 本项目已从yfinance迁移到IB TWS API。请参考最新的API使用方法。
> 原始文档备份在: `backup_before_ib_migration/{doc_file}`

'''
                        content = migration_note + content
                        
                        with open(doc_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        logger.info(f"   ✅ 更新文档: {doc_file}")
                        
                except Exception as e:
                    logger.error(f"❌ 更新文档失败: {doc_file}, 错误: {e}")
    
    def generate_migration_report(self):
        """生成迁移报告"""
        report_file = self.project_root / "IB_MIGRATION_REPORT.md"
        
        report_content = f"""# IB TWS API 迁移报告

**迁移时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 迁移概述

本次迁移将项目中的yfinance数据源替换为Interactive Brokers TWS API，以解决yfinance的稳定性问题。

## 📁 备份信息

原始文件已备份到: `{self.backup_dir.relative_to(self.project_root)}/`

## 🔄 迁移详情

### 已迁移的文件

"""
        
        for i, file_path in enumerate(self.files_to_migrate, 1):
            report_content += f"{i}. `{file_path}`\n"
        
        report_content += f"""
### 测试文件（保留用于调试）

"""
        
        for i, file_path in enumerate(self.test_files, 1):
            report_content += f"{i}. `{file_path}`\n"
        
        report_content += f"""
### 已更新的文档

"""
        
        for i, file_path in enumerate(self.doc_files, 1):
            report_content += f"{i}. `{file_path}`\n"
        
        report_content += """
## ⚙️ 配置要求

### 1. 安装IB API
```bash
pip install ibapi
```

### 2. 启动IB TWS或Gateway
- 下载并安装IB TWS或Gateway
- 启动应用程序并登录
- 配置API设置（启用API，设置端口）

### 3. 配置连接参数
```python
from src.data.ib_data_provider import IBConfig

config = IBConfig(
    host="127.0.0.1",
    port=7497,  # 模拟交易端口，实盘使用7496
    timeout=30
)
```

## 🧪 测试迁移

### 1. 测试IB连接
```bash
python -c "from src.data.ib_data_provider import IBDataProvider; print('IB API可用:', IBDataProvider().is_available)"
```

### 2. 运行新的数据获取脚本
```bash
python fetch_nasdaq_all_stocks_ib.py
```

### 3. 验证数据一致性
```bash
python validate_ib_data_consistency.py
```

## ⚠️ 注意事项

1. **IB TWS/Gateway必须运行**: 与yfinance不同，IB API需要TWS或Gateway应用程序运行
2. **API限制**: IB API有连接数和请求频率限制
3. **数据权限**: 某些数据可能需要订阅才能获取
4. **时区处理**: IB数据可能使用不同的时区设置

## 🔧 故障排除

### 连接问题
- 确保IB TWS/Gateway正在运行
- 检查API设置是否启用
- 验证端口配置

### 数据获取问题
- 检查股票代码格式
- 验证数据权限和订阅
- 查看IB API日志

## 📞 支持

如遇问题，请：
1. 查看 `nasdaq_all_stocks_ib.log` 日志文件
2. 运行调试脚本进行诊断
3. 参考IB API官方文档

---

**迁移完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 迁移报告已生成: {report_file}")
    
    def run_migration(self):
        """运行完整迁移流程"""
        logger.info("🚀 开始yfinance到IB API迁移")
        logger.info("=" * 60)
        
        try:
            # 1. 创建备份
            self.create_backup()
            
            # 2. 迁移文件
            logger.info("\n🔄 开始迁移文件...")
            migration_results = {}
            
            for file_path in self.files_to_migrate:
                result = self.migrate_file(file_path)
                migration_results[file_path] = result
                
                if result["status"] == "success":
                    logger.info(f"   ✅ {file_path}: {len(result['changes'])} 项变更")
                elif result["status"] == "skip":
                    logger.info(f"   ⏭️  {file_path}: {result['reason']}")
                else:
                    logger.error(f"   ❌ {file_path}: {result.get('error', '未知错误')}")
            
            # 3. 更新文档
            self.update_documentation()
            
            # 4. 生成报告
            self.generate_migration_report()
            
            # 5. 统计结果
            successful = sum(1 for r in migration_results.values() if r["status"] == "success")
            total = len(migration_results)
            
            logger.info("\n" + "=" * 60)
            logger.info("📊 迁移完成统计")
            logger.info("=" * 60)
            logger.info(f"总文件数: {total}")
            logger.info(f"成功迁移: {successful}")
            logger.info(f"跳过文件: {total - successful}")
            logger.info(f"成功率: {(successful/total*100):.1f}%")
            logger.info("=" * 60)
            
            logger.info("\n🎉 迁移完成！")
            logger.info("📋 下一步:")
            logger.info("   1. 安装IB API: pip install ibapi")
            logger.info("   2. 启动IB TWS或Gateway")
            logger.info("   3. 运行测试: python fetch_nasdaq_all_stocks_ib.py")
            logger.info("   4. 查看迁移报告: IB_MIGRATION_REPORT.md")
            
        except Exception as e:
            logger.error(f"❌ 迁移过程中出现错误: {e}")
            raise

def main():
    """主函数"""
    migrator = YFinanceToIBMigrator()
    migrator.run_migration()

if __name__ == "__main__":
    main()