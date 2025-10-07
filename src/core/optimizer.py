"""
优化器模块

提供贝叶斯优化、遗传算法和多目标优化功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BayesianOptimizer:
    """贝叶斯优化器"""
    
    def __init__(self, 
                 acquisition_function: str = 'expected_improvement',
                 n_initial_points: int = 10,
                 random_state: int = 42):
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.optimization_history = []
        self.best_params = None
        self.best_score = None
        
    def optimize(self, 
                objective_function: Callable,
                parameter_bounds: Dict[str, Tuple[float, float]],
                n_calls: int = 100,
                **kwargs) -> Dict[str, Any]:
        """执行贝叶斯优化"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
            
            # 定义搜索空间
            dimensions = []
            param_names = []
            
            for param_name, (low, high) in parameter_bounds.items():
                dimensions.append(Real(low, high, name=param_name))
                param_names.append(param_name)
            
            # 包装目标函数
            @use_named_args(dimensions)
            def wrapped_objective(**params):
                try:
                    score = objective_function(params)
                    self.optimization_history.append({
                        'params': params.copy(),
                        'score': score
                    })
                    return -score  # skopt最小化，所以取负值
                except Exception as e:
                    print(f"Error in objective function: {e}")
                    return 1e6  # 返回大的惩罚值
            
            # 执行优化
            result = gp_minimize(
                func=wrapped_objective,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=self.n_initial_points,
                acq_func=self.acquisition_function,
                random_state=self.random_state,
                **kwargs
            )
            
            # 保存最佳结果
            self.best_params = dict(zip(param_names, result.x))
            self.best_score = -result.fun
            
            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_result': result,
                'history': self.optimization_history
            }
            
        except ImportError:
            print("scikit-optimize not available, using random search")
            return self._random_search(objective_function, parameter_bounds, n_calls)
    
    def _random_search(self, 
                      objective_function: Callable,
                      parameter_bounds: Dict[str, Tuple[float, float]],
                      n_calls: int) -> Dict[str, Any]:
        """随机搜索作为后备方案"""
        np.random.seed(self.random_state)
        
        best_score = -np.inf
        best_params = None
        
        for _ in range(n_calls):
            # 随机采样参数
            params = {}
            for param_name, (low, high) in parameter_bounds.items():
                params[param_name] = np.random.uniform(low, high)
            
            # 评估目标函数
            try:
                score = objective_function(params)
                self.optimization_history.append({
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                print(f"Error in objective function: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'history': self.optimization_history
        }


class GeneticOptimizer:
    """遗传算法优化器"""
    
    def __init__(self,
                 population_size: int = 50,
                 n_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 random_state: int = 42):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state
        self.optimization_history = []
        self.best_params = None
        self.best_score = None
        
    def optimize(self,
                objective_function: Callable,
                parameter_bounds: Dict[str, Tuple[float, float]],
                **kwargs) -> Dict[str, Any]:
        """执行遗传算法优化"""
        np.random.seed(self.random_state)
        
        param_names = list(parameter_bounds.keys())
        bounds = np.array(list(parameter_bounds.values()))
        
        # 初始化种群
        population = self._initialize_population(bounds)
        
        best_score_history = []
        avg_score_history = []
        
        for generation in range(self.n_generations):
            # 评估种群
            scores = self._evaluate_population(population, param_names, objective_function)
            
            # 记录历史
            best_idx = np.argmax(scores)
            best_score_history.append(scores[best_idx])
            avg_score_history.append(np.mean(scores))
            
            # 更新最佳结果
            if self.best_score is None or scores[best_idx] > self.best_score:
                self.best_score = scores[best_idx]
                self.best_params = dict(zip(param_names, population[best_idx]))
            
            # 选择
            selected_population = self._selection(population, scores)
            
            # 交叉
            offspring = self._crossover(selected_population)
            
            # 变异
            offspring = self._mutation(offspring, bounds)
            
            # 更新种群
            population = offspring
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Score = {best_score_history[-1]:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_score_history': best_score_history,
            'avg_score_history': avg_score_history,
            'history': self.optimization_history
        }
    
    def _initialize_population(self, bounds: np.ndarray) -> np.ndarray:
        """初始化种群"""
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            (self.population_size, len(bounds))
        )
        return population
    
    def _evaluate_population(self, 
                           population: np.ndarray,
                           param_names: List[str],
                           objective_function: Callable) -> np.ndarray:
        """评估种群适应度"""
        scores = []
        
        for individual in population:
            params = dict(zip(param_names, individual))
            try:
                score = objective_function(params)
                scores.append(score)
                self.optimization_history.append({
                    'params': params.copy(),
                    'score': score
                })
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                scores.append(-np.inf)
        
        return np.array(scores)
    
    def _selection(self, population: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """锦标赛选择"""
        selected = []
        
        for _ in range(self.population_size):
            # 锦标赛选择
            tournament_size = 3
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            tournament_scores = scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            selected.append(population[winner_idx].copy())
        
        return np.array(selected)
    
    def _crossover(self, population: np.ndarray) -> np.ndarray:
        """交叉操作"""
        offspring = []
        
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]
            
            if np.random.random() < self.crossover_rate:
                # 算术交叉
                alpha = np.random.random()
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = (1 - alpha) * parent1 + alpha * parent2
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return np.array(offspring[:self.population_size])
    
    def _mutation(self, population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """变异操作"""
        for individual in population:
            for i in range(len(individual)):
                if np.random.random() < self.mutation_rate:
                    # 高斯变异
                    mutation_strength = 0.1 * (bounds[i, 1] - bounds[i, 0])
                    individual[i] += np.random.normal(0, mutation_strength)
                    
                    # 边界处理
                    individual[i] = np.clip(individual[i], bounds[i, 0], bounds[i, 1])
        
        return population


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, 
                 algorithm: str = 'nsga2',
                 population_size: int = 100,
                 n_generations: int = 200,
                 random_state: int = 42):
        self.algorithm = algorithm
        self.population_size = population_size
        self.n_generations = n_generations
        self.random_state = random_state
        self.pareto_front = None
        self.optimization_history = []
        
    def optimize(self,
                objective_functions: List[Callable],
                parameter_bounds: Dict[str, Tuple[float, float]],
                objective_names: Optional[List[str]] = None,
                **kwargs) -> Dict[str, Any]:
        """执行多目标优化"""
        if objective_names is None:
            objective_names = [f'objective_{i}' for i in range(len(objective_functions))]
        
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize
            from pymoo.core.variable import Real
            
            # 定义优化问题
            class MultiObjectiveProblem(Problem):
                def __init__(self):
                    param_names = list(parameter_bounds.keys())
                    bounds = list(parameter_bounds.values())
                    
                    super().__init__(
                        n_var=len(param_names),
                        n_obj=len(objective_functions),
                        xl=np.array([b[0] for b in bounds]),
                        xu=np.array([b[1] for b in bounds])
                    )
                    self.param_names = param_names
                
                def _evaluate(self, X, out, *args, **kwargs):
                    objectives = []
                    
                    for x in X:
                        params = dict(zip(self.param_names, x))
                        obj_values = []
                        
                        for obj_func in objective_functions:
                            try:
                                value = obj_func(params)
                                obj_values.append(-value)  # 最小化，所以取负值
                            except Exception as e:
                                print(f"Error in objective function: {e}")
                                obj_values.append(1e6)
                        
                        objectives.append(obj_values)
                        self.optimization_history.append({
                            'params': params.copy(),
                            'objectives': dict(zip(objective_names, [-v for v in obj_values]))
                        })
                    
                    out["F"] = np.array(objectives)
            
            # 创建问题实例
            problem = MultiObjectiveProblem()
            
            # 创建算法
            if self.algorithm == 'nsga2':
                algorithm = NSGA2(pop_size=self.population_size)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            # 执行优化
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.n_generations),
                seed=self.random_state,
                verbose=False
            )
            
            # 提取帕累托前沿
            pareto_solutions = []
            for i, x in enumerate(result.X):
                params = dict(zip(problem.param_names, x))
                objectives = dict(zip(objective_names, -result.F[i]))  # 转回最大化
                pareto_solutions.append({
                    'params': params,
                    'objectives': objectives
                })
            
            self.pareto_front = pareto_solutions
            
            return {
                'pareto_front': self.pareto_front,
                'optimization_result': result,
                'history': self.optimization_history
            }
            
        except ImportError:
            print("pymoo not available, using weighted sum approach")
            return self._weighted_sum_optimization(
                objective_functions, parameter_bounds, objective_names
            )
    
    def _weighted_sum_optimization(self,
                                 objective_functions: List[Callable],
                                 parameter_bounds: Dict[str, Tuple[float, float]],
                                 objective_names: List[str]) -> Dict[str, Any]:
        """加权和方法作为后备方案"""
        # 使用不同权重组合进行多次单目标优化
        weight_combinations = [
            [1.0, 0.0], [0.8, 0.2], [0.6, 0.4], [0.5, 0.5],
            [0.4, 0.6], [0.2, 0.8], [0.0, 1.0]
        ]
        
        if len(objective_functions) > 2:
            # 对于多于2个目标的情况，使用均匀权重
            n_objectives = len(objective_functions)
            weight_combinations = []
            for i in range(10):  # 生成10个不同的权重组合
                weights = np.random.dirichlet(np.ones(n_objectives))
                weight_combinations.append(weights.tolist())
        
        pareto_solutions = []
        
        for weights in weight_combinations:
            # 定义加权目标函数
            def weighted_objective(params):
                total_score = 0
                for i, obj_func in enumerate(objective_functions):
                    try:
                        score = obj_func(params)
                        total_score += weights[i] * score
                    except Exception as e:
                        print(f"Error in objective function {i}: {e}")
                        return -1e6
                return total_score
            
            # 使用贝叶斯优化求解
            optimizer = BayesianOptimizer(random_state=self.random_state)
            result = optimizer.optimize(
                weighted_objective, parameter_bounds, n_calls=50
            )
            
            if result['best_params'] is not None:
                # 计算所有目标值
                objectives = {}
                for i, obj_func in enumerate(objective_functions):
                    try:
                        objectives[objective_names[i]] = obj_func(result['best_params'])
                    except Exception as e:
                        objectives[objective_names[i]] = -1e6
                
                pareto_solutions.append({
                    'params': result['best_params'],
                    'objectives': objectives,
                    'weights': weights
                })
        
        self.pareto_front = pareto_solutions
        
        return {
            'pareto_front': self.pareto_front,
            'history': self.optimization_history
        }
    
    def get_best_compromise_solution(self, 
                                   preference_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """获取最佳折中解"""
        if self.pareto_front is None:
            raise ValueError("No Pareto front available. Run optimization first.")
        
        if preference_weights is None:
            # 使用等权重
            objective_names = list(self.pareto_front[0]['objectives'].keys())
            preference_weights = {name: 1.0/len(objective_names) for name in objective_names}
        
        best_score = -np.inf
        best_solution = None
        
        for solution in self.pareto_front:
            # 计算加权得分
            weighted_score = sum(
                preference_weights.get(obj_name, 0) * obj_value
                for obj_name, obj_value in solution['objectives'].items()
            )
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_solution = solution
        
        return best_solution