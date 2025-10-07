#!/usr/bin/env python3
"""
比赛环境配置脚本
用于快速设置和切换不同量化比赛的环境配置
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class CompetitionSetup:
    """比赛环境配置管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "configs" / "competitions"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 比赛配置模板
        self.competition_configs = {
            "jane_street": {
                "name": "Jane Street Market Prediction",
                "type": "classification",
                "data_format": "tabular",
                "target_column": "action",
                "evaluation_metric": "weighted_log_loss",
                "time_series": True,
                "features": {
                    "numerical_features": 130,
                    "categorical_features": 0,
                    "time_features": ["date_id", "ts_id"]
                },
                "model_config": {
                    "primary_models": ["lightgbm", "xgboost", "catboost"],
                    "ensemble_method": "stacking",
                    "cv_strategy": "time_series_split",
                    "cv_folds": 5
                },
                "feature_engineering": {
                    "lag_features": [1, 2, 3, 5, 10],
                    "rolling_windows": [5, 10, 20, 50],
                    "interaction_features": True,
                    "target_encoding": True
                },
                "optimization": {
                    "objective": "minimize_log_loss",
                    "early_stopping": 100,
                    "max_iterations": 10000
                }
            },
            
            "optiver": {
                "name": "Optiver - Trading at the Close",
                "type": "regression",
                "data_format": "order_book",
                "target_column": "target",
                "evaluation_metric": "rmse",
                "time_series": True,
                "features": {
                    "price_features": ["bid_price", "ask_price", "wap"],
                    "volume_features": ["bid_size", "ask_size", "matched_size"],
                    "time_features": ["seconds_in_bucket", "time_id"]
                },
                "model_config": {
                    "primary_models": ["lightgbm", "lstm", "transformer"],
                    "ensemble_method": "weighted_average",
                    "cv_strategy": "time_series_split",
                    "cv_folds": 5
                },
                "feature_engineering": {
                    "microstructure_features": True,
                    "order_flow_features": True,
                    "volatility_features": True,
                    "rolling_windows": [10, 30, 60, 300]
                },
                "optimization": {
                    "objective": "minimize_rmse",
                    "real_time_constraints": True,
                    "max_inference_time": 100  # milliseconds
                }
            },
            
            "citadel": {
                "name": "Citadel Terminal AI Competition",
                "type": "trading",
                "data_format": "real_time",
                "target_column": "returns",
                "evaluation_metric": "sharpe_ratio",
                "time_series": True,
                "features": {
                    "price_features": ["open", "high", "low", "close", "volume"],
                    "technical_indicators": ["rsi", "macd", "bollinger", "ma"],
                    "market_features": ["volatility", "momentum", "mean_reversion"]
                },
                "model_config": {
                    "strategies": ["momentum", "mean_reversion", "arbitrage", "ml_prediction"],
                    "ensemble_method": "dynamic_weighting",
                    "rebalance_frequency": "minute"
                },
                "risk_management": {
                    "max_position_size": 0.1,
                    "max_portfolio_var": 0.02,
                    "stop_loss": 0.05,
                    "max_drawdown": 0.15
                },
                "optimization": {
                    "objective": "maximize_sharpe",
                    "constraints": ["risk_limits", "position_limits"],
                    "real_time_execution": True
                }
            },
            
            "cme_crypto": {
                "name": "CME Group Crypto Classic",
                "type": "crypto_trading",
                "data_format": "futures_options",
                "target_column": "pnl",
                "evaluation_metric": "total_return",
                "time_series": True,
                "features": {
                    "futures_features": ["basis", "contango", "backwardation"],
                    "options_features": ["implied_vol", "delta", "gamma", "theta"],
                    "crypto_features": ["funding_rate", "open_interest", "volume"]
                },
                "model_config": {
                    "strategies": ["basis_trading", "calendar_spread", "volatility_trading"],
                    "instruments": ["BTC_futures", "ETH_futures", "BTC_options", "ETH_options"],
                    "rebalance_frequency": "daily"
                },
                "risk_management": {
                    "max_leverage": 3.0,
                    "position_limits": {"BTC": 0.4, "ETH": 0.3},
                    "var_limit": 0.05,
                    "correlation_limit": 0.7
                },
                "optimization": {
                    "objective": "maximize_return",
                    "constraints": ["leverage_limits", "position_limits"],
                    "account_size": 100000
                }
            }
        }
    
    def create_competition_config(self, competition: str) -> None:
        """创建比赛配置文件"""
        if competition not in self.competition_configs:
            raise ValueError(f"Unknown competition: {competition}")
        
        config = self.competition_configs[competition]
        config_file = self.config_dir / f"{competition}_config.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created config for {config['name']}")
        print(f"📁 Config file: {config_file}")
    
    def setup_competition_environment(self, competition: str) -> None:
        """设置比赛环境"""
        print(f"🚀 Setting up environment for {competition}...")
        
        # 1. 创建配置文件
        self.create_competition_config(competition)
        
        # 2. 创建比赛专用目录
        comp_dir = self.project_root / "competitions" / competition
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        subdirs = ["data", "models", "features", "submissions", "logs"]
        for subdir in subdirs:
            (comp_dir / subdir).mkdir(exist_ok=True)
        
        # 3. 创建比赛专用脚本
        self._create_competition_scripts(competition, comp_dir)
        
        # 4. 创建环境变量文件
        self._create_env_file(competition, comp_dir)
        
        print(f"✅ Environment setup complete for {competition}")
        print(f"📁 Competition directory: {comp_dir}")
    
    def _create_competition_scripts(self, competition: str, comp_dir: Path) -> None:
        """创建比赛专用脚本"""
        config = self.competition_configs[competition]
        
        # 训练脚本
        train_script = f"""#!/usr/bin/env python3
\"\"\"
{config['name']} 训练脚本
\"\"\"

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import DataLoader
from src.features.factor_calculator import FactorCalculator
from src.ml.advanced_feature_engineering import AdvancedFeatureEngineer
from src.ml.model_ensemble import ModelEnsemble
from src.ml.model_validation import TimeSeriesValidator

def main():
    print("🚀 Starting {config['name']} training...")
    
    # 加载数据
    data_loader = DataLoader()
    # TODO: 加载比赛数据
    
    # 特征工程
    feature_engineer = AdvancedFeatureEngineer()
    # TODO: 构建特征
    
    # 模型训练
    ensemble = ModelEnsemble()
    # TODO: 训练模型
    
    # 验证
    validator = TimeSeriesValidator()
    # TODO: 验证模型
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
"""
        
        with open(comp_dir / "train.py", 'w') as f:
            f.write(train_script)
        
        # 预测脚本
        predict_script = f"""#!/usr/bin/env python3
\"\"\"
{config['name']} 预测脚本
\"\"\"

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main():
    print("🔮 Starting {config['name']} prediction...")
    
    # TODO: 加载模型和数据
    # TODO: 生成预测
    # TODO: 创建提交文件
    
    print("✅ Prediction completed!")

if __name__ == "__main__":
    main()
"""
        
        with open(comp_dir / "predict.py", 'w') as f:
            f.write(predict_script)
        
        # 使脚本可执行
        os.chmod(comp_dir / "train.py", 0o755)
        os.chmod(comp_dir / "predict.py", 0o755)
    
    def _create_env_file(self, competition: str, comp_dir: Path) -> None:
        """创建环境变量文件"""
        config = self.competition_configs[competition]
        
        env_content = f"""# {config['name']} 环境配置

# 比赛基本信息
COMPETITION_NAME="{config['name']}"
COMPETITION_TYPE="{config['type']}"
EVALUATION_METRIC="{config['evaluation_metric']}"

# 数据路径
DATA_DIR="{comp_dir}/data"
MODEL_DIR="{comp_dir}/models"
FEATURE_DIR="{comp_dir}/features"
SUBMISSION_DIR="{comp_dir}/submissions"
LOG_DIR="{comp_dir}/logs"

# 模型配置
PRIMARY_MODELS="{','.join(config['model_config'].get('primary_models', []))}"
ENSEMBLE_METHOD="{config['model_config'].get('ensemble_method', 'stacking')}"
CV_STRATEGY="{config['model_config'].get('cv_strategy', 'time_series_split')}"
CV_FOLDS="{config['model_config'].get('cv_folds', 5)}"

# 特征工程配置
TIME_SERIES="{config.get('time_series', True)}"
TARGET_COLUMN="{config.get('target_column', 'target')}"

# 优化配置
OBJECTIVE="{config['optimization'].get('objective', 'minimize_loss')}"
"""
        
        with open(comp_dir / ".env", 'w') as f:
            f.write(env_content)
    
    def list_competitions(self) -> None:
        """列出所有可用的比赛"""
        print("📋 Available competitions:")
        for key, config in self.competition_configs.items():
            print(f"  • {key}: {config['name']}")
            print(f"    Type: {config['type']}")
            print(f"    Metric: {config['evaluation_metric']}")
            print()
    
    def setup_all_competitions(self) -> None:
        """设置所有比赛环境"""
        print("🚀 Setting up all competition environments...")
        
        for competition in self.competition_configs.keys():
            try:
                self.setup_competition_environment(competition)
                print()
            except Exception as e:
                print(f"❌ Failed to setup {competition}: {e}")
                print()
        
        print("✅ All competition environments setup complete!")
    
    def create_master_script(self) -> None:
        """创建主控制脚本"""
        master_script = """#!/usr/bin/env python3
\"\"\"
量化比赛主控制脚本
\"\"\"

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.competition_setup import CompetitionSetup

def main():
    parser = argparse.ArgumentParser(description="量化比赛环境管理")
    parser.add_argument("action", choices=["setup", "list", "train", "predict"], 
                       help="执行的操作")
    parser.add_argument("--competition", "-c", 
                       choices=["jane_street", "optiver", "citadel", "cme_crypto", "all"],
                       help="比赛名称")
    
    args = parser.parse_args()
    
    setup = CompetitionSetup()
    
    if args.action == "list":
        setup.list_competitions()
    elif args.action == "setup":
        if args.competition == "all":
            setup.setup_all_competitions()
        elif args.competition:
            setup.setup_competition_environment(args.competition)
        else:
            print("❌ Please specify a competition with --competition")
    elif args.action == "train":
        if args.competition and args.competition != "all":
            comp_dir = Path(__file__).parent.parent / "competitions" / args.competition
            train_script = comp_dir / "train.py"
            if train_script.exists():
                import subprocess
                subprocess.run([sys.executable, str(train_script)])
            else:
                print(f"❌ Training script not found: {train_script}")
        else:
            print("❌ Please specify a competition with --competition")
    elif args.action == "predict":
        if args.competition and args.competition != "all":
            comp_dir = Path(__file__).parent.parent / "competitions" / args.competition
            predict_script = comp_dir / "predict.py"
            if predict_script.exists():
                import subprocess
                subprocess.run([sys.executable, str(predict_script)])
            else:
                print(f"❌ Prediction script not found: {predict_script}")
        else:
            print("❌ Please specify a competition with --competition")

if __name__ == "__main__":
    main()
"""
        
        master_file = self.project_root / "scripts" / "competition_manager.py"
        with open(master_file, 'w') as f:
            f.write(master_script)
        
        os.chmod(master_file, 0o755)
        print(f"✅ Created master script: {master_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="量化比赛环境配置")
    parser.add_argument("action", choices=["setup", "list", "all", "master"], 
                       help="执行的操作")
    parser.add_argument("--competition", "-c", 
                       choices=["jane_street", "optiver", "citadel", "cme_crypto"],
                       help="比赛名称")
    
    args = parser.parse_args()
    
    setup = CompetitionSetup()
    
    if args.action == "list":
        setup.list_competitions()
    elif args.action == "setup":
        if args.competition:
            setup.setup_competition_environment(args.competition)
        else:
            print("❌ Please specify a competition with --competition")
    elif args.action == "all":
        setup.setup_all_competitions()
    elif args.action == "master":
        setup.create_master_script()

if __name__ == "__main__":
    main()