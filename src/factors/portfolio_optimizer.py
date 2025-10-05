"""
组合优化器

实现多种组合优化算法和约束处理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy import optimize
from scipy.linalg import sqrtm
import cvxpy as cp
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraint:
    """优化约束"""
    name: str
    constraint_type: str  # 'equality', 'inequality', 'bound'
    constraint_func: Callable
    bounds: Optional[Tuple[float, float]] = None
    tolerance: float = 1e-6
    active: bool = True
    
    def evaluate(self, weights: np.ndarray) -> float:
        """评估约束"""
        return self.constraint_func(weights)


@dataclass
class OptimizationObjective:
    """优化目标"""
    name: str
    objective_func: Callable
    weight: float = 1.0
    maximize: bool = True  # True为最大化，False为最小化
    
    def evaluate(self, weights: np.ndarray) -> float:
        """评估目标函数"""
        value = self.objective_func(weights)
        return value if self.maximize else -value


@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    weights: pd.Series
    objective_value: float
    constraint_violations: Dict[str, float]
    optimization_time: float
    iterations: int
    message: str
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """获取组合指标"""
        return {
            'total_weight': self.weights.sum(),
            'max_weight': self.weights.max(),
            'min_weight': self.weights.min(),
            'concentration': (self.weights ** 2).sum(),
            'effective_stocks': 1 / (self.weights ** 2).sum(),
            'long_exposure': self.weights[self.weights > 0].sum(),
            'short_exposure': abs(self.weights[self.weights < 0].sum()),
            'net_exposure': self.weights.sum(),
            'gross_exposure': abs(self.weights).sum()
        }


class BaseOptimizer(ABC):
    """
    基础优化器抽象类
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger
        
        # 约束和目标
        self.constraints: List[OptimizationConstraint] = []
        self.objectives: List[OptimizationObjective] = []
        
        # 优化历史
        self.optimization_history: List[OptimizationResult] = []
    
    def add_constraint(self, constraint: OptimizationConstraint):
        """添加约束"""
        self.constraints.append(constraint)
        self.logger.info(f"已添加约束: {constraint.name}")
    
    def add_objective(self, objective: OptimizationObjective):
        """添加目标函数"""
        self.objectives.append(objective)
        self.logger.info(f"已添加目标: {objective.name}")
    
    def remove_constraint(self, constraint_name: str):
        """移除约束"""
        self.constraints = [c for c in self.constraints if c.name != constraint_name]
        self.logger.info(f"已移除约束: {constraint_name}")
    
    def remove_objective(self, objective_name: str):
        """移除目标函数"""
        self.objectives = [o for o in self.objectives if o.name != objective_name]
        self.logger.info(f"已移除目标: {objective_name}")
    
    @abstractmethod
    def optimize(self, expected_returns: pd.Series, 
                covariance_matrix: pd.DataFrame,
                **kwargs) -> OptimizationResult:
        """执行优化"""
        pass
    
    def validate_inputs(self, expected_returns: pd.Series, 
                       covariance_matrix: pd.DataFrame):
        """验证输入数据"""
        if len(expected_returns) != len(covariance_matrix):
            raise ValueError("预期收益率和协方差矩阵维度不匹配")
        
        if not expected_returns.index.equals(covariance_matrix.index):
            raise ValueError("预期收益率和协方差矩阵索引不匹配")
        
        if not covariance_matrix.index.equals(covariance_matrix.columns):
            raise ValueError("协方差矩阵不是方阵")
        
        # 检查协方差矩阵是否正定
        eigenvals = np.linalg.eigvals(covariance_matrix.values)
        if np.any(eigenvals <= 0):
            self.logger.warning("协方差矩阵不是正定的，将进行修正")


class MeanVarianceOptimizer(BaseOptimizer):
    """
    均值-方差优化器
    
    实现经典的Markowitz均值-方差优化
    """
    
    def __init__(self, risk_aversion: float = 1.0):
        """
        初始化均值-方差优化器
        
        Args:
            risk_aversion: 风险厌恶系数
        """
        super().__init__("MeanVariance")
        self.risk_aversion = risk_aversion
    
    def optimize(self, expected_returns: pd.Series, 
                covariance_matrix: pd.DataFrame,
                initial_weights: Optional[pd.Series] = None,
                **kwargs) -> OptimizationResult:
        """
        执行均值-方差优化
        
        Args:
            expected_returns: 预期收益率
            covariance_matrix: 协方差矩阵
            initial_weights: 初始权重
            
        Returns:
            OptimizationResult: 优化结果
        """
        import time
        start_time = time.time()
        
        self.validate_inputs(expected_returns, covariance_matrix)
        
        n_assets = len(expected_returns)
        
        # 初始权重
        if initial_weights is None:
            x0 = np.ones(n_assets) / n_assets
        else:
            x0 = initial_weights.reindex(expected_returns.index, fill_value=0).values
        
        # 目标函数：最大化 μ'w - λ/2 * w'Σw
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns.values)
            portfolio_risk = np.dot(weights, np.dot(covariance_matrix.values, weights))
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_risk)
        
        # 约束条件
        constraints = []
        
        # 权重和为1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # 添加自定义约束
        for constraint in self.constraints:
            if constraint.active:
                if constraint.constraint_type == 'equality':
                    constraints.append({
                        'type': 'eq',
                        'fun': constraint.constraint_func
                    })
                elif constraint.constraint_type == 'inequality':
                    constraints.append({
                        'type': 'ineq',
                        'fun': constraint.constraint_func
                    })
        
        # 权重边界
        bounds = []
        for constraint in self.constraints:
            if constraint.constraint_type == 'bound' and constraint.bounds:
                bounds.extend([constraint.bounds] * n_assets)
        
        if not bounds:
            bounds = [(0, 1)] * n_assets  # 默认边界
        
        # 执行优化
        try:
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            optimization_time = time.time() - start_time
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
                
                # 检查约束违反
                constraint_violations = {}
                for constraint in self.constraints:
                    if constraint.active:
                        violation = abs(constraint.evaluate(result.x))
                        if violation > constraint.tolerance:
                            constraint_violations[constraint.name] = violation
                
                optimization_result = OptimizationResult(
                    success=True,
                    weights=optimal_weights,
                    objective_value=-result.fun,
                    constraint_violations=constraint_violations,
                    optimization_time=optimization_time,
                    iterations=result.nit,
                    message=result.message
                )
                
            else:
                optimization_result = OptimizationResult(
                    success=False,
                    weights=pd.Series(x0, index=expected_returns.index),
                    objective_value=float('inf'),
                    constraint_violations={},
                    optimization_time=optimization_time,
                    iterations=result.nit,
                    message=result.message
                )
            
            self.optimization_history.append(optimization_result)
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"优化失败: {e}")
            return OptimizationResult(
                success=False,
                weights=pd.Series(x0, index=expected_returns.index),
                objective_value=float('inf'),
                constraint_violations={},
                optimization_time=time.time() - start_time,
                iterations=0,
                message=str(e)
            )


class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman优化器
    
    实现Black-Litterman模型的组合优化
    """
    
    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.025):
        """
        初始化Black-Litterman优化器
        
        Args:
            risk_aversion: 风险厌恶系数
            tau: 不确定性参数
        """
        super().__init__("BlackLitterman")
        self.risk_aversion = risk_aversion
        self.tau = tau
    
    def optimize(self, expected_returns: pd.Series, 
                covariance_matrix: pd.DataFrame,
                market_cap_weights: pd.Series,
                views_matrix: Optional[pd.DataFrame] = None,
                views_returns: Optional[pd.Series] = None,
                views_uncertainty: Optional[pd.DataFrame] = None,
                **kwargs) -> OptimizationResult:
        """
        执行Black-Litterman优化
        
        Args:
            expected_returns: 预期收益率（先验）
            covariance_matrix: 协方差矩阵
            market_cap_weights: 市值权重
            views_matrix: 观点矩阵P
            views_returns: 观点收益率Q
            views_uncertainty: 观点不确定性矩阵Ω
            
        Returns:
            OptimizationResult: 优化结果
        """
        import time
        start_time = time.time()
        
        self.validate_inputs(expected_returns, covariance_matrix)
        
        # 计算隐含收益率（先验均值）
        implied_returns = self.risk_aversion * covariance_matrix @ market_cap_weights
        
        # 如果没有观点，使用隐含收益率
        if views_matrix is None or views_returns is None:
            bl_returns = implied_returns
            bl_covariance = covariance_matrix
        else:
            # Black-Litterman公式
            # 先验协方差矩阵
            prior_covariance = self.tau * covariance_matrix
            
            # 观点不确定性矩阵
            if views_uncertainty is None:
                # 使用对角矩阵，不确定性与先验方差成比例
                views_uncertainty = pd.DataFrame(
                    np.diag(np.diag(views_matrix @ prior_covariance @ views_matrix.T)),
                    index=views_matrix.index,
                    columns=views_matrix.index
                )
            
            # 计算后验均值和协方差
            P = views_matrix.values
            Q = views_returns.values
            Omega = views_uncertainty.values
            
            # 后验协方差矩阵
            M1 = np.linalg.inv(prior_covariance.values)
            M2 = P.T @ np.linalg.inv(Omega) @ P
            bl_covariance_inv = M1 + M2
            bl_covariance = pd.DataFrame(
                np.linalg.inv(bl_covariance_inv),
                index=covariance_matrix.index,
                columns=covariance_matrix.columns
            )
            
            # 后验均值
            mu1 = M1 @ implied_returns.values
            mu2 = P.T @ np.linalg.inv(Omega) @ Q
            bl_returns = pd.Series(
                bl_covariance.values @ (mu1 + mu2),
                index=expected_returns.index
            )
        
        # 使用均值-方差优化器求解
        mv_optimizer = MeanVarianceOptimizer(self.risk_aversion)
        
        # 复制约束
        for constraint in self.constraints:
            mv_optimizer.add_constraint(constraint)
        
        result = mv_optimizer.optimize(bl_returns, bl_covariance, **kwargs)
        
        # 更新结果信息
        result.optimization_time = time.time() - start_time
        
        self.optimization_history.append(result)
        return result


class RiskParityOptimizer(BaseOptimizer):
    """
    风险平价优化器
    
    实现风险平价组合优化
    """
    
    def __init__(self, risk_budget: Optional[pd.Series] = None):
        """
        初始化风险平价优化器
        
        Args:
            risk_budget: 风险预算，如果为None则使用等风险预算
        """
        super().__init__("RiskParity")
        self.risk_budget = risk_budget
    
    def optimize(self, expected_returns: pd.Series, 
                covariance_matrix: pd.DataFrame,
                **kwargs) -> OptimizationResult:
        """
        执行风险平价优化
        
        Args:
            expected_returns: 预期收益率
            covariance_matrix: 协方差矩阵
            
        Returns:
            OptimizationResult: 优化结果
        """
        import time
        start_time = time.time()
        
        self.validate_inputs(expected_returns, covariance_matrix)
        
        n_assets = len(expected_returns)
        
        # 风险预算
        if self.risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets
        else:
            risk_budget = self.risk_budget.reindex(expected_returns.index, fill_value=0).values
        
        # 目标函数：最小化风险贡献与目标风险预算的偏差
        def objective(weights):
            # 计算风险贡献
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix.values, weights))
            marginal_contrib = np.dot(covariance_matrix.values, weights)
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            # 计算与目标风险预算的偏差
            deviation = risk_contrib - risk_budget
            return np.sum(deviation ** 2)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # 权重和为1
        ]
        
        # 权重边界
        bounds = [(0.001, 1)] * n_assets  # 避免零权重
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            optimization_time = time.time() - start_time
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
                
                optimization_result = OptimizationResult(
                    success=True,
                    weights=optimal_weights,
                    objective_value=result.fun,
                    constraint_violations={},
                    optimization_time=optimization_time,
                    iterations=result.nit,
                    message=result.message
                )
            else:
                optimization_result = OptimizationResult(
                    success=False,
                    weights=pd.Series(x0, index=expected_returns.index),
                    objective_value=float('inf'),
                    constraint_violations={},
                    optimization_time=optimization_time,
                    iterations=result.nit,
                    message=result.message
                )
            
            self.optimization_history.append(optimization_result)
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"风险平价优化失败: {e}")
            return OptimizationResult(
                success=False,
                weights=pd.Series(x0, index=expected_returns.index),
                objective_value=float('inf'),
                constraint_violations={},
                optimization_time=time.time() - start_time,
                iterations=0,
                message=str(e)
            )


class CVXPYOptimizer(BaseOptimizer):
    """
    CVXPY优化器
    
    使用CVXPY库实现凸优化
    """
    
    def __init__(self):
        super().__init__("CVXPY")
    
    def optimize(self, expected_returns: pd.Series, 
                covariance_matrix: pd.DataFrame,
                risk_aversion: float = 1.0,
                **kwargs) -> OptimizationResult:
        """
        使用CVXPY执行优化
        
        Args:
            expected_returns: 预期收益率
            covariance_matrix: 协方差矩阵
            risk_aversion: 风险厌恶系数
            
        Returns:
            OptimizationResult: 优化结果
        """
        import time
        start_time = time.time()
        
        self.validate_inputs(expected_returns, covariance_matrix)
        
        n_assets = len(expected_returns)
        
        # 定义优化变量
        weights = cp.Variable(n_assets)
        
        # 目标函数：最大化效用 = 预期收益 - 风险惩罚
        portfolio_return = expected_returns.values.T @ weights
        portfolio_risk = cp.quad_form(weights, covariance_matrix.values)
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_risk
        
        # 约束条件
        constraints = [
            cp.sum(weights) == 1,  # 权重和为1
            weights >= 0           # 权重非负
        ]
        
        # 添加自定义约束
        for constraint in self.constraints:
            if constraint.active:
                if constraint.constraint_type == 'bound' and constraint.bounds:
                    lower, upper = constraint.bounds
                    constraints.extend([
                        weights >= lower,
                        weights <= upper
                    ])
        
        # 定义问题
        problem = cp.Problem(cp.Maximize(utility), constraints)
        
        try:
            # 求解
            problem.solve(solver=cp.ECOS, verbose=False)
            
            optimization_time = time.time() - start_time
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = pd.Series(weights.value, index=expected_returns.index)
                
                optimization_result = OptimizationResult(
                    success=True,
                    weights=optimal_weights,
                    objective_value=problem.value,
                    constraint_violations={},
                    optimization_time=optimization_time,
                    iterations=0,  # CVXPY不提供迭代次数
                    message="Optimal solution found"
                )
            else:
                optimization_result = OptimizationResult(
                    success=False,
                    weights=pd.Series(np.ones(n_assets) / n_assets, index=expected_returns.index),
                    objective_value=float('inf'),
                    constraint_violations={},
                    optimization_time=optimization_time,
                    iterations=0,
                    message=f"Optimization failed: {problem.status}"
                )
            
            self.optimization_history.append(optimization_result)
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"CVXPY优化失败: {e}")
            return OptimizationResult(
                success=False,
                weights=pd.Series(np.ones(n_assets) / n_assets, index=expected_returns.index),
                objective_value=float('inf'),
                constraint_violations={},
                optimization_time=time.time() - start_time,
                iterations=0,
                message=str(e)
            )


class PortfolioOptimizer:
    """
    组合优化器管理类
    
    统一管理多种优化算法
    """
    
    def __init__(self):
        self.optimizers: Dict[str, BaseOptimizer] = {}
        self.logger = logger
        
        # 注册默认优化器
        self.register_optimizer("mean_variance", MeanVarianceOptimizer())
        self.register_optimizer("black_litterman", BlackLittermanOptimizer())
        self.register_optimizer("risk_parity", RiskParityOptimizer())
        
        try:
            self.register_optimizer("cvxpy", CVXPYOptimizer())
        except ImportError:
            self.logger.warning("CVXPY未安装，跳过CVXPY优化器")
    
    def register_optimizer(self, name: str, optimizer: BaseOptimizer):
        """注册优化器"""
        self.optimizers[name] = optimizer
        self.logger.info(f"已注册优化器: {name}")
    
    def get_optimizer(self, name: str) -> BaseOptimizer:
        """获取优化器"""
        if name not in self.optimizers:
            raise ValueError(f"未找到优化器: {name}")
        return self.optimizers[name]
    
    def list_optimizers(self) -> List[str]:
        """列出所有可用的优化器"""
        return list(self.optimizers.keys())
    
    def optimize(self, optimizer_name: str, 
                expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                **kwargs) -> OptimizationResult:
        """
        执行组合优化
        
        Args:
            optimizer_name: 优化器名称
            expected_returns: 预期收益率
            covariance_matrix: 协方差矩阵
            **kwargs: 其他参数
            
        Returns:
            OptimizationResult: 优化结果
        """
        optimizer = self.get_optimizer(optimizer_name)
        return optimizer.optimize(expected_returns, covariance_matrix, **kwargs)
    
    def compare_optimizers(self, expected_returns: pd.Series,
                          covariance_matrix: pd.DataFrame,
                          optimizer_names: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, OptimizationResult]:
        """
        比较多个优化器的结果
        
        Args:
            expected_returns: 预期收益率
            covariance_matrix: 协方差矩阵
            optimizer_names: 要比较的优化器名称列表
            **kwargs: 其他参数
            
        Returns:
            Dict[str, OptimizationResult]: 各优化器的结果
        """
        if optimizer_names is None:
            optimizer_names = self.list_optimizers()
        
        results = {}
        
        for name in optimizer_names:
            try:
                self.logger.info(f"运行优化器: {name}")
                result = self.optimize(name, expected_returns, covariance_matrix, **kwargs)
                results[name] = result
                
                if result.success:
                    self.logger.info(f"{name} 优化成功，目标值: {result.objective_value:.6f}")
                else:
                    self.logger.warning(f"{name} 优化失败: {result.message}")
                    
            except Exception as e:
                self.logger.error(f"{name} 优化异常: {e}")
                results[name] = OptimizationResult(
                    success=False,
                    weights=pd.Series(np.ones(len(expected_returns)) / len(expected_returns), 
                                    index=expected_returns.index),
                    objective_value=float('inf'),
                    constraint_violations={},
                    optimization_time=0,
                    iterations=0,
                    message=str(e)
                )
        
        return results
    
    def create_efficient_frontier(self, expected_returns: pd.Series,
                                 covariance_matrix: pd.DataFrame,
                                 n_points: int = 50) -> pd.DataFrame:
        """
        创建有效前沿
        
        Args:
            expected_returns: 预期收益率
            covariance_matrix: 协方差矩阵
            n_points: 前沿点数
            
        Returns:
            pd.DataFrame: 有效前沿数据
        """
        # 计算最小方差组合
        min_var_optimizer = MeanVarianceOptimizer(risk_aversion=1e6)  # 极高风险厌恶
        min_var_result = min_var_optimizer.optimize(expected_returns, covariance_matrix)
        
        if not min_var_result.success:
            raise ValueError("无法计算最小方差组合")
        
        min_var_weights = min_var_result.weights
        min_return = expected_returns @ min_var_weights
        min_risk = np.sqrt(min_var_weights @ covariance_matrix @ min_var_weights)
        
        # 计算最大收益组合（忽略风险）
        max_ret_optimizer = MeanVarianceOptimizer(risk_aversion=1e-6)  # 极低风险厌恶
        max_ret_result = max_ret_optimizer.optimize(expected_returns, covariance_matrix)
        
        if not max_ret_result.success:
            raise ValueError("无法计算最大收益组合")
        
        max_ret_weights = max_ret_result.weights
        max_return = expected_returns @ max_ret_weights
        max_risk = np.sqrt(max_ret_weights @ covariance_matrix @ max_ret_weights)
        
        # 生成有效前沿
        target_returns = np.linspace(min_return, max_return, n_points)
        frontier_data = []
        
        for target_return in target_returns:
            # 添加收益率约束
            return_constraint = OptimizationConstraint(
                name="target_return",
                constraint_type="equality",
                constraint_func=lambda w, tr=target_return: expected_returns.values @ w - tr
            )
            
            # 最小化方差
            optimizer = MeanVarianceOptimizer(risk_aversion=1e6)
            optimizer.add_constraint(return_constraint)
            
            result = optimizer.optimize(expected_returns, covariance_matrix)
            
            if result.success:
                portfolio_return = expected_returns @ result.weights
                portfolio_risk = np.sqrt(result.weights @ covariance_matrix @ result.weights)
                
                frontier_data.append({
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
                    'weights': result.weights.to_dict()
                })
        
        return pd.DataFrame(frontier_data)


# 常用约束函数
def weight_sum_constraint(weights: np.ndarray) -> float:
    """权重和约束"""
    return np.sum(weights) - 1.0


def long_only_constraint(weights: np.ndarray) -> np.ndarray:
    """多头约束"""
    return weights


def sector_constraint(weights: np.ndarray, sector_mapping: Dict[int, str], 
                     max_sector_weight: float = 0.3) -> np.ndarray:
    """行业约束"""
    sector_weights = {}
    for i, weight in enumerate(weights):
        sector = sector_mapping.get(i, 'Unknown')
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    violations = []
    for sector, weight in sector_weights.items():
        if weight > max_sector_weight:
            violations.append(weight - max_sector_weight)
    
    return np.array(violations) if violations else np.array([0])


def turnover_constraint(weights: np.ndarray, previous_weights: np.ndarray,
                       max_turnover: float = 0.1) -> float:
    """换手率约束"""
    turnover = np.sum(np.abs(weights - previous_weights)) / 2
    return max_turnover - turnover


if __name__ == "__main__":
    # 测试组合优化器
    
    # 创建测试数据
    n_assets = 5
    np.random.seed(42)
    
    expected_returns = pd.Series(
        np.random.normal(0.08, 0.02, n_assets),
        index=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # 生成随机协方差矩阵
    A = np.random.randn(n_assets, n_assets)
    covariance_matrix = pd.DataFrame(
        A @ A.T / n_assets,
        index=expected_returns.index,
        columns=expected_returns.index
    )
    
    # 创建优化器管理器
    optimizer_manager = PortfolioOptimizer()
    
    print("可用优化器:", optimizer_manager.list_optimizers())
    
    # 比较不同优化器
    results = optimizer_manager.compare_optimizers(
        expected_returns, covariance_matrix
    )
    
    for name, result in results.items():
        if result.success:
            metrics = result.get_portfolio_metrics()
            print(f"\n{name} 优化结果:")
            print(f"  目标值: {result.objective_value:.6f}")
            print(f"  最大权重: {metrics['max_weight']:.4f}")
            print(f"  有效股票数: {metrics['effective_stocks']:.2f}")
        else:
            print(f"\n{name} 优化失败: {result.message}")