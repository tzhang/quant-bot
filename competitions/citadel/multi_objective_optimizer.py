#!/usr/bin/env python3
"""
多目标优化器 - 用于Citadel策略的高级参数优化
同时优化收益率、夏普比率、最大回撤等多个目标
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 多目标优化相关库
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import optuna
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, strategy_evaluator):
        self.strategy_evaluator = strategy_evaluator
        self.optimization_history = []
        self.pareto_front = None
        self.best_solutions = {}
        
        # 优化目标权重
        self.objectives = {
            'total_return': {'weight': 0.3, 'direction': 'maximize'},
            'sharpe_ratio': {'weight': 0.3, 'direction': 'maximize'},
            'max_drawdown': {'weight': 0.2, 'direction': 'minimize'},
            'win_rate': {'weight': 0.1, 'direction': 'maximize'},
            'volatility': {'weight': 0.1, 'direction': 'minimize'}
        }
        
        # 参数边界
        self.parameter_bounds = {
            'signal_threshold': (0.01, 0.15),
            'volume_threshold': (0.8, 3.0),
            'volatility_threshold': (0.005, 0.08),
            'momentum_weight': (0.1, 0.8),
            'mean_reversion_weight': (0.1, 0.6),
            'microstructure_weight': (0.05, 0.3),
            'volume_weight': (0.05, 0.3),
            'technical_weight': (0.05, 0.3),
            'stop_loss': (0.005, 0.08),
            'take_profit': (0.02, 0.15),
            'trailing_stop': (0.005, 0.05),
            'max_position_size': (0.1, 0.8),
            'lookback_period': (5, 50)
        }
    
    class StrategyOptimizationProblem(Problem):
        """策略优化问题定义"""
        
        def __init__(self, optimizer):
            self.optimizer = optimizer
            
            # 定义变量数量和边界
            n_vars = len(optimizer.parameter_bounds)
            xl = np.array([bounds[0] for bounds in optimizer.parameter_bounds.values()])
            xu = np.array([bounds[1] for bounds in optimizer.parameter_bounds.values()])
            
            super().__init__(n_var=n_vars, n_obj=5, xl=xl, xu=xu)
        
        def _evaluate(self, X, out, *args, **kwargs):
            """评估函数"""
            objectives = []
            
            for x in X:
                # 构建参数字典
                params = dict(zip(self.optimizer.parameter_bounds.keys(), x))
                
                # 评估策略性能
                metrics = self.optimizer.strategy_evaluator.evaluate_parameters(params)
                
                # 构建目标函数值（注意：pymoo最小化所有目标，所以需要转换）
                obj_values = [
                    -metrics.get('total_return', 0),  # 最大化收益率 -> 最小化负收益率
                    -metrics.get('sharpe_ratio', 0),  # 最大化夏普比率
                    metrics.get('max_drawdown', 100),  # 最小化最大回撤
                    -metrics.get('win_rate', 0),      # 最大化胜率
                    metrics.get('volatility', 100)    # 最小化波动率
                ]
                
                objectives.append(obj_values)
            
            out["F"] = np.array(objectives)
    
    def nsga2_optimization(self, n_generations=50, population_size=100):
        """使用NSGA-II进行多目标优化"""
        print("🎯 执行NSGA-II多目标优化...")
        
        # 定义优化问题
        problem = self.StrategyOptimizationProblem(self)
        
        # 配置NSGA-II算法
        algorithm = NSGA2(
            pop_size=population_size,
            eliminate_duplicates=True
        )
        
        # 执行优化
        result = minimize(
            problem,
            algorithm,
            ('n_gen', n_generations),
            verbose=True
        )
        
        # 提取帕累托前沿
        self.pareto_front = result.F
        pareto_params = result.X
        
        # 转换回参数字典
        pareto_solutions = []
        for i, params_array in enumerate(pareto_params):
            params = dict(zip(self.parameter_bounds.keys(), params_array))
            objectives = {
                'total_return': -self.pareto_front[i][0],
                'sharpe_ratio': -self.pareto_front[i][1],
                'max_drawdown': self.pareto_front[i][2],
                'win_rate': -self.pareto_front[i][3],
                'volatility': self.pareto_front[i][4]
            }
            pareto_solutions.append({
                'parameters': params,
                'objectives': objectives,
                'rank': i
            })
        
        print(f"✅ NSGA-II优化完成，找到 {len(pareto_solutions)} 个帕累托最优解")
        
        return pareto_solutions
    
    def bayesian_multi_objective_optimization(self, n_trials=100):
        """贝叶斯多目标优化"""
        print("🧠 执行贝叶斯多目标优化...")
        
        # 创建多个单目标优化研究
        studies = {}
        
        for objective_name in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
            direction = 'maximize' if self.objectives[objective_name]['direction'] == 'maximize' else 'minimize'
            
            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(),
                study_name=f"optimize_{objective_name}"
            )
            
            def objective_func(trial, target_objective=objective_name):
                # 定义参数
                params = {}
                for param_name, (low, high) in self.parameter_bounds.items():
                    if isinstance(low, int) and isinstance(high, int):
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high)
                
                # 评估策略
                metrics = self.strategy_evaluator.evaluate_parameters(params)
                return metrics.get(target_objective, 0)
            
            study.optimize(objective_func, n_trials=n_trials//4, show_progress_bar=True)
            studies[objective_name] = study
        
        # 收集所有最优解
        bayesian_solutions = []
        for objective_name, study in studies.items():
            best_params = study.best_params
            best_value = study.best_value
            
            # 重新评估以获取所有指标
            all_metrics = self.strategy_evaluator.evaluate_parameters(best_params)
            
            bayesian_solutions.append({
                'parameters': best_params,
                'objectives': all_metrics,
                'primary_objective': objective_name,
                'primary_value': best_value
            })
        
        print(f"✅ 贝叶斯多目标优化完成，找到 {len(bayesian_solutions)} 个专门优化解")
        
        return bayesian_solutions
    
    def gaussian_process_optimization(self, n_iterations=50):
        """高斯过程多目标优化"""
        print("📊 执行高斯过程多目标优化...")
        
        # 初始化高斯过程模型
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp_models = {}
        
        for objective in self.objectives.keys():
            gp_models[objective] = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
        
        # 初始采样
        n_initial = 20
        initial_samples = []
        initial_objectives = []
        
        for _ in range(n_initial):
            # 随机采样参数
            params = {}
            for param_name, (low, high) in self.parameter_bounds.items():
                params[param_name] = np.random.uniform(low, high)
            
            # 评估
            metrics = self.strategy_evaluator.evaluate_parameters(params)
            
            initial_samples.append(list(params.values()))
            initial_objectives.append(metrics)
        
        X_samples = np.array(initial_samples)
        
        # 训练初始GP模型
        for objective in self.objectives.keys():
            y_values = np.array([obj.get(objective, 0) for obj in initial_objectives])
            gp_models[objective].fit(X_samples, y_values)
        
        # 迭代优化
        best_solutions = []
        
        for iteration in range(n_iterations):
            # 使用采集函数选择下一个评估点
            next_params = self._acquisition_function_optimization(gp_models)
            
            # 评估新点
            metrics = self.strategy_evaluator.evaluate_parameters(next_params)
            
            # 更新数据集
            X_samples = np.vstack([X_samples, list(next_params.values())])
            initial_objectives.append(metrics)
            
            # 重新训练GP模型
            for objective in self.objectives.keys():
                y_values = np.array([obj.get(objective, 0) for obj in initial_objectives])
                gp_models[objective].fit(X_samples, y_values)
            
            # 记录当前最优解
            best_solutions.append({
                'parameters': next_params,
                'objectives': metrics,
                'iteration': iteration
            })
            
            if iteration % 10 == 0:
                print(f"   迭代 {iteration}/{n_iterations} 完成")
        
        print(f"✅ 高斯过程优化完成，评估了 {len(best_solutions)} 个解")
        
        return best_solutions
    
    def _acquisition_function_optimization(self, gp_models):
        """采集函数优化"""
        def acquisition_function(x):
            # 期望改进 (Expected Improvement)
            total_ei = 0
            
            for objective, gp_model in gp_models.items():
                x_reshaped = x.reshape(1, -1)
                mu, sigma = gp_model.predict(x_reshaped, return_std=True)
                
                # 当前最优值
                current_best = np.max([obj.get(objective, 0) for obj in self.optimization_history])
                
                # 计算期望改进
                if sigma > 0:
                    z = (mu - current_best) / sigma
                    ei = sigma * (z * self._normal_cdf(z) + self._normal_pdf(z))
                else:
                    ei = 0
                
                weight = self.objectives[objective]['weight']
                total_ei += weight * ei[0]
            
            return -total_ei  # 最小化负期望改进
        
        # 优化采集函数
        bounds = list(self.parameter_bounds.values())
        result = differential_evolution(acquisition_function, bounds, maxiter=100)
        
        # 转换回参数字典
        optimal_params = dict(zip(self.parameter_bounds.keys(), result.x))
        return optimal_params
    
    def _normal_cdf(self, x):
        """标准正态分布累积分布函数"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _normal_pdf(self, x):
        """标准正态分布概率密度函数"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def ensemble_optimization(self, n_trials_per_method=30):
        """集成多种优化方法"""
        print("🚀 执行集成多目标优化...")
        
        all_solutions = []
        
        # 1. NSGA-II优化
        try:
            nsga2_solutions = self.nsga2_optimization(n_generations=20, population_size=50)
            all_solutions.extend(nsga2_solutions)
            print(f"   NSGA-II贡献 {len(nsga2_solutions)} 个解")
        except Exception as e:
            print(f"   ⚠️  NSGA-II优化失败: {e}")
        
        # 2. 贝叶斯优化
        try:
            bayesian_solutions = self.bayesian_multi_objective_optimization(n_trials=n_trials_per_method)
            all_solutions.extend(bayesian_solutions)
            print(f"   贝叶斯优化贡献 {len(bayesian_solutions)} 个解")
        except Exception as e:
            print(f"   ⚠️  贝叶斯优化失败: {e}")
        
        # 3. 高斯过程优化
        try:
            gp_solutions = self.gaussian_process_optimization(n_iterations=20)
            all_solutions.extend(gp_solutions)
            print(f"   高斯过程优化贡献 {len(gp_solutions)} 个解")
        except Exception as e:
            print(f"   ⚠️  高斯过程优化失败: {e}")
        
        # 分析和排序解
        ranked_solutions = self.rank_solutions(all_solutions)
        
        print(f"\n✅ 集成优化完成，共找到 {len(all_solutions)} 个解")
        print(f"   经过排序后的前10个解:")
        
        for i, solution in enumerate(ranked_solutions[:10]):
            objectives = solution['objectives']
            print(f"   {i+1:2d}. 收益率: {objectives.get('total_return', 0):.2f}% | "
                  f"夏普: {objectives.get('sharpe_ratio', 0):.2f} | "
                  f"回撤: {objectives.get('max_drawdown', 0):.2f}% | "
                  f"胜率: {objectives.get('win_rate', 0):.1f}%")
        
        return ranked_solutions
    
    def rank_solutions(self, solutions):
        """对解进行排序"""
        # 计算综合得分
        for solution in solutions:
            objectives = solution['objectives']
            
            # 标准化目标值
            score = 0
            for obj_name, obj_config in self.objectives.items():
                value = objectives.get(obj_name, 0)
                weight = obj_config['weight']
                
                # 根据优化方向调整得分
                if obj_config['direction'] == 'maximize':
                    score += weight * value
                else:
                    score -= weight * value
            
            solution['composite_score'] = score
        
        # 按综合得分排序
        return sorted(solutions, key=lambda x: x['composite_score'], reverse=True)
    
    def visualize_optimization_results(self, solutions):
        """可视化优化结果"""
        if not solutions:
            print("⚠️  没有优化结果可供可视化")
            return
        
        # 提取目标值
        objectives_data = {}
        for obj_name in self.objectives.keys():
            objectives_data[obj_name] = [sol['objectives'].get(obj_name, 0) for sol in solutions]
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('多目标优化结果分析', fontsize=16, fontweight='bold')
        
        # 1. 目标值分布
        axes[0, 0].hist(objectives_data['total_return'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('收益率分布')
        axes[0, 0].set_xlabel('收益率 (%)')
        
        axes[0, 1].hist(objectives_data['sharpe_ratio'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('夏普比率分布')
        axes[0, 1].set_xlabel('夏普比率')
        
        axes[0, 2].hist(objectives_data['max_drawdown'], bins=20, alpha=0.7, color='red')
        axes[0, 2].set_title('最大回撤分布')
        axes[0, 2].set_xlabel('最大回撤 (%)')
        
        # 2. 目标间相关性
        axes[1, 0].scatter(objectives_data['total_return'], objectives_data['sharpe_ratio'], alpha=0.6)
        axes[1, 0].set_xlabel('收益率 (%)')
        axes[1, 0].set_ylabel('夏普比率')
        axes[1, 0].set_title('收益率 vs 夏普比率')
        
        axes[1, 1].scatter(objectives_data['total_return'], objectives_data['max_drawdown'], alpha=0.6, color='orange')
        axes[1, 1].set_xlabel('收益率 (%)')
        axes[1, 1].set_ylabel('最大回撤 (%)')
        axes[1, 1].set_title('收益率 vs 最大回撤')
        
        # 3. 帕累托前沿（如果有的话）
        if len(solutions) > 10:
            # 选择前20%的解作为近似帕累托前沿
            top_solutions = solutions[:len(solutions)//5]
            pareto_returns = [sol['objectives'].get('total_return', 0) for sol in top_solutions]
            pareto_sharpe = [sol['objectives'].get('sharpe_ratio', 0) for sol in top_solutions]
            
            axes[1, 2].scatter(objectives_data['total_return'], objectives_data['sharpe_ratio'], 
                             alpha=0.3, color='lightblue', label='所有解')
            axes[1, 2].scatter(pareto_returns, pareto_sharpe, 
                             alpha=0.8, color='red', s=50, label='帕累托前沿')
            axes[1, 2].set_xlabel('收益率 (%)')
            axes[1, 2].set_ylabel('夏普比率')
            axes[1, 2].set_title('帕累托前沿')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('/tmp/multi_objective_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 优化结果可视化已保存至: /tmp/multi_objective_optimization_results.png")
    
    def get_best_solutions_by_objective(self, solutions, top_k=3):
        """按不同目标获取最优解"""
        best_solutions = {}
        
        for obj_name, obj_config in self.objectives.items():
            # 按特定目标排序
            if obj_config['direction'] == 'maximize':
                sorted_solutions = sorted(solutions, 
                                        key=lambda x: x['objectives'].get(obj_name, 0), 
                                        reverse=True)
            else:
                sorted_solutions = sorted(solutions, 
                                        key=lambda x: x['objectives'].get(obj_name, float('inf')))
            
            best_solutions[obj_name] = sorted_solutions[:top_k]
        
        return best_solutions
    
    def export_optimization_results(self, solutions, filename='optimization_results.csv'):
        """导出优化结果"""
        if not solutions:
            print("⚠️  没有结果可导出")
            return
        
        # 准备数据
        export_data = []
        for i, solution in enumerate(solutions):
            row = {'solution_id': i}
            
            # 添加参数
            for param_name, param_value in solution['parameters'].items():
                row[f'param_{param_name}'] = param_value
            
            # 添加目标值
            for obj_name, obj_value in solution['objectives'].items():
                row[f'obj_{obj_name}'] = obj_value
            
            # 添加综合得分
            row['composite_score'] = solution.get('composite_score', 0)
            
            export_data.append(row)
        
        # 创建DataFrame并导出
        df = pd.DataFrame(export_data)
        filepath = f'/tmp/{filename}'
        df.to_csv(filepath, index=False)
        
        print(f"📁 优化结果已导出至: {filepath}")
        print(f"   共导出 {len(solutions)} 个解，{len(df.columns)} 个特征")

def main():
    """主函数 - 演示多目标优化"""
    print("🎯 多目标优化器演示")
    print("=" * 60)
    
    # 模拟策略评估器
    class MockStrategyEvaluator:
        def evaluate_parameters(self, params):
            # 模拟评估结果
            np.random.seed(hash(str(params)) % 2**32)
            
            return {
                'total_return': np.random.normal(15, 10),
                'sharpe_ratio': np.random.normal(1.5, 0.8),
                'max_drawdown': abs(np.random.normal(5, 3)),
                'win_rate': np.random.uniform(40, 70),
                'volatility': abs(np.random.normal(12, 4))
            }
    
    # 创建优化器
    evaluator = MockStrategyEvaluator()
    optimizer = MultiObjectiveOptimizer(evaluator)
    
    # 执行集成优化
    solutions = optimizer.ensemble_optimization(n_trials_per_method=20)
    
    # 可视化结果
    optimizer.visualize_optimization_results(solutions)
    
    # 获取各目标的最优解
    best_by_objective = optimizer.get_best_solutions_by_objective(solutions)
    
    print("\n🏆 各目标最优解:")
    for obj_name, best_sols in best_by_objective.items():
        print(f"\n{obj_name} 最优解:")
        for i, sol in enumerate(best_sols[:3]):
            obj_val = sol['objectives'][obj_name]
            print(f"  {i+1}. {obj_name}: {obj_val:.3f}")
    
    # 导出结果
    optimizer.export_optimization_results(solutions)
    
    print("\n🚀 多目标优化演示完成!")

if __name__ == "__main__":
    main()