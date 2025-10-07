#!/usr/bin/env python3
"""
量化比赛主控制脚本
"""

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
