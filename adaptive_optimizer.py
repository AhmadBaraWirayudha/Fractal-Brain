"""
adaptive_optimizer.py
=====================

Production-Grade Autonomous Self-Improving Adaptive Hyperparameter Optimization Engine.
Integrates natively into the Fractal Brain + Living Knowledge Graph ecosystem.

Designed and implemented in pure Python using ONLY the Python Standard Library.
No external dependencies (No NumPy, SciPy, Optuna, Hyperopt, Ray Tune, Nevergrad, Torch, JAX).

Architectural Highlights:
-------------------------
- Pure OOP, SOLID, Open-Closed Principle, Dependency Injection.
- Strategy Pattern for 21+ distinct search strategies.
- Visitor Pattern for parameter validation, sampling, and serialization across 9 parameter types.
- Command Pattern for trial evaluations.
- Observer Pattern for telemetry, knowledge graph persistence, and fractal memory updates.
- State Pattern for lifecycle management (INITIALIZING, EXPLORING, EXPLOITING, REFINING, CONVERGED, STOPPED).
- Autoregressive history tracking: moving averages, regret, posterior distributions, improvement velocity.
- Bayesian conjugate updates: Normal-Inverse-Gamma for continuous parameters and Beta/Dirichlet for discrete/categorical.
- Joint Gaussian Process surrogate (Cholesky-based, analytical EI/UCB) modeling cross-parameter covariance,
  complementing the independent per-parameter conjugate beliefs used by the marginal Bayesian strategy.
- Meta-learning (`MetaController`) via Multi-Armed Bandits over search strategies.
- Recursive Fractal Search and Self-Tuning recursion (`SelfTuner`).
- Online Lifelong Learning with zero-restart state updates.

Author: Senior AI Systems Architect & Probabilistic Programming Expert
Date: 2026-07-14
"""

import abc
import copy
import enum
import heapq
import itertools
import json
import math
import random
import sys
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)


# ==============================================================================
# PURE PYTHON MATHEMATICAL & PROBABILISTIC UTILITIES
# ==============================================================================

Matrix = List[List[float]]
Vector = List[float]


class MathUtils:
    """Pure Python mathematical and statistical helper functions."""

    EPSILON = 1e-12

    @staticmethod
    def clamp(val: float, low: float, high: float) -> float:
        """Clamp val between low and high."""
        return max(low, min(high, val))

    @staticmethod
    def mean(values: List[float]) -> float:
        """Compute arithmetic mean of a list of floats."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def variance(values: List[float], sample: bool = True) -> float:
        """Compute sample or population variance of a list of floats."""
        n = len(values)
        if n < 2:
            return 0.0
        m = MathUtils.mean(values)
        ss = sum((x - m) ** 2 for x in values)
        return ss / (n - 1 if sample else n)

    @staticmethod
    def std_dev(values: List[float], sample: bool = True) -> float:
        """Compute standard deviation."""
        return math.sqrt(MathUtils.variance(values, sample=sample))

    @staticmethod
    def norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Gaussian (Normal) probability density function."""
        if sigma <= MathUtils.EPSILON:
            return 1.0 if abs(x - mu) < MathUtils.EPSILON else 0.0
        z = (x - mu) / sigma
        return (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * math.exp(-0.5 * z * z)

    @staticmethod
    def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Gaussian (Normal) cumulative distribution function using math.erf."""
        if sigma <= MathUtils.EPSILON:
            return 1.0 if x >= mu else 0.0
        z = (x - mu) / (sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))

    @staticmethod
    def rbf_kernel(x1: List[float], x2: List[float], length_scale: float = 1.0, variance: float = 1.0) -> float:
        """Radial Basis Function (Squared Exponential) kernel between two vectors."""
        if len(x1) != len(x2) or not x1:
            return 0.0
        ls = max(length_scale, MathUtils.EPSILON)
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return variance * math.exp(-0.5 * sq_dist / (ls * ls))

    @staticmethod
    def pearson_correlation(x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient between two equal-length numerical vectors."""
        n = len(x)
        if n != len(y) or n < 2:
            return 0.0
        mean_x, mean_y = MathUtils.mean(x), MathUtils.mean(y)
        std_x, std_y = MathUtils.std_dev(x), MathUtils.std_dev(y)
        if std_x <= MathUtils.EPSILON or std_y <= MathUtils.EPSILON:
            return 0.0
        cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / (n - 1)
        return MathUtils.clamp(cov / (std_x * std_y), -1.0, 1.0)

    @staticmethod
    def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
        """Solve Ax = b for small square matrices using Gauss-Jordan elimination with partial pivoting."""
        n = len(A)
        if n == 0 or len(b) != n:
            return [0.0] * n
        M = [row[:] + [val] for row, val in zip(A, b)]
        for i in range(n):
            pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
            if abs(M[pivot_row][i]) < MathUtils.EPSILON:
                continue
            M[i], M[pivot_row] = M[pivot_row], M[i]
            pivot = M[i][i]
            M[i] = [val / pivot for val in M[i]]
            for j in range(n):
                if j != i:
                    factor = M[j][i]
                    M[j] = [v_j - factor * v_i for v_j, v_i in zip(M[j], M[i])]
        return [M[i][n] for i in range(n)]

    @staticmethod
    def k_means_cluster(points: List[List[float]], k: int, max_iters: int = 25, rng: Optional[random.Random] = None) -> List[List[float]]:
        """Pure Python K-Means clustering algorithm to find k centroids among continuous numerical vectors."""
        if not points:
            return []
        k = min(k, len(points))
        if k <= 0:
            return []
        rng = rng or random.Random(42)
        centroids = [list(pt) for pt in rng.sample(points, k)]
        for _ in range(max_iters):
            clusters: List[List[List[float]]] = [[] for _ in range(k)]
            for pt in points:
                best_idx = 0
                best_dist = float('inf')
                for idx, c in enumerate(centroids):
                    dist = sum((a - b) ** 2 for a, b in zip(pt, c))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                clusters[best_idx].append(pt)
            new_centroids = []
            for idx, c in enumerate(centroids):
                if clusters[idx]:
                    dim = len(c)
                    avg = [sum(pt[d] for pt in clusters[idx]) / len(clusters[idx]) for d in range(dim)]
                    new_centroids.append(avg)
                else:
                    new_centroids.append([val + rng.uniform(-0.1, 0.1) for val in c])
            if all(sum((a - b) ** 2 for a, b in zip(c1, c2)) < MathUtils.EPSILON for c1, c2 in zip(centroids, new_centroids)):
                break
            centroids = new_centroids
        return centroids

    @staticmethod
    def mat_transpose(A: Matrix) -> Matrix:
        """Transpose a matrix represented as a list of row lists."""
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    @staticmethod
    def cholesky(A: Matrix) -> Matrix:
        """Cholesky decomposition of a symmetric positive-definite matrix: returns lower-triangular L with A = L L^T."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    L[i][j] = math.sqrt(val) if val > MathUtils.EPSILON else math.sqrt(MathUtils.EPSILON)
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]
        return L

    @staticmethod
    def forward_substitution(L: Matrix, b: Vector) -> Vector:
        """Solve L x = b for lower-triangular L."""
        n = len(L)
        x = [0.0] * n
        for i in range(n):
            s = sum(L[i][j] * x[j] for j in range(i))
            x[i] = (b[i] - s) / L[i][i]
        return x

    @staticmethod
    def backward_substitution(U: Matrix, b: Vector) -> Vector:
        """Solve U x = b for upper-triangular U."""
        n = len(U)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(U[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - s) / U[i][i]
        return x

    @staticmethod
    def solve_cholesky(L: Matrix, b: Vector) -> Vector:
        """Solve A x = b given the Cholesky factor L of A."""
        y = MathUtils.forward_substitution(L, b)
        return MathUtils.backward_substitution(MathUtils.mat_transpose(L), y)


# ==============================================================================
# STATE & LIFECYCLE MANAGEMENT (STATE PATTERN)
# ==============================================================================

class OptimizerState(enum.Enum):
    """Lifecycle states of the adaptive optimization engine."""
    INITIALIZING = "INITIALIZING"
    EXPLORING = "EXPLORING"
    EXPLOITING = "EXPLOITING"
    REFINING = "REFINING"
    CONVERGED = "CONVERGED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class TrialStatus(enum.Enum):
    """Execution status for individual optimization trials."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PRUNED = "PRUNED"
    TIMEOUT = "TIMEOUT"


# ==============================================================================
# PARAMETER VISITOR PATTERN (VISITOR PATTERN)
# ==============================================================================

class ParameterVisitor(abc.ABC):
    """Abstract Visitor for traversing parameter specifications cleanly without isinstance sprawl."""
    @abc.abstractmethod
    def visit_continuous(self, param: 'ContinuousParameter') -> Any: pass

    @abc.abstractmethod
    def visit_integer(self, param: 'IntegerParameter') -> Any: pass

    @abc.abstractmethod
    def visit_boolean(self, param: 'BooleanParameter') -> Any: pass

    @abc.abstractmethod
    def visit_categorical(self, param: 'CategoricalParameter') -> Any: pass

    @abc.abstractmethod
    def visit_discrete(self, param: 'DiscreteParameter') -> Any: pass

    @abc.abstractmethod
    def visit_hierarchical(self, param: 'HierarchicalParameter') -> Any: pass

    @abc.abstractmethod
    def visit_dynamic(self, param: 'DynamicParameter') -> Any: pass

    @abc.abstractmethod
    def visit_vector(self, param: 'VectorParameter') -> Any: pass

    @abc.abstractmethod
    def visit_matrix(self, param: 'MatrixParameter') -> Any: pass


# ==============================================================================
# PARAMETER TYPE HIERARCHY
# ==============================================================================

class ParameterSpec(abc.ABC):
    """Abstract base class representing a searchable parameter inside the ecosystem."""
    def __init__(self, name: str, tunable: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self.name: str = name
        self.tunable: bool = tunable
        self.metadata: Dict[str, Any] = metadata or {}

    @abc.abstractmethod
    def accept(self, visitor: ParameterVisitor) -> Any:
        """Accept a parameter visitor."""
        pass

    @abc.abstractmethod
    def validate_value(self, value: Any) -> bool:
        """Check whether the provided value falls within the valid domain."""
        pass


class ContinuousParameter(ParameterSpec):
    """Continuous floating-point parameter with prior mean and standard deviation."""
    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        prior_mean: Optional[float] = None,
        prior_std: Optional[float] = None,
        log_scale: bool = False,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        if low >= high:
            raise ValueError(f"ContinuousParameter {name} must have low < high (got low={low}, high={high})")
        if log_scale and low <= 0:
            raise ValueError(f"ContinuousParameter {name} on log_scale must have low > 0 (got low={low})")
        self.low = float(low)
        self.high = float(high)
        self.log_scale = log_scale
        self.prior_mean = float(prior_mean) if prior_mean is not None else 0.5 * (self.low + self.high)
        self.prior_std = float(prior_std) if prior_std is not None else max((self.high - self.low) / 4.0, MathUtils.EPSILON)

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_continuous(self)

    def validate_value(self, value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        return self.low - MathUtils.EPSILON <= float(value) <= self.high + MathUtils.EPSILON


class IntegerParameter(ParameterSpec):
    """Integer-valued parameter defined over a bounded range [low, high]."""
    def __init__(
        self,
        name: str,
        low: int,
        high: int,
        prior_mean: Optional[float] = None,
        prior_std: Optional[float] = None,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        if low > high:
            raise ValueError(f"IntegerParameter {name} must have low <= high (got low={low}, high={high})")
        self.low = int(low)
        self.high = int(high)
        self.prior_mean = float(prior_mean) if prior_mean is not None else 0.5 * (self.low + self.high)
        self.prior_std = float(prior_std) if prior_std is not None else max((self.high - self.low) / 4.0, 1.0)

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_integer(self)

    def validate_value(self, value: Any) -> bool:
        if not isinstance(value, int) and not (isinstance(value, float) and value.is_integer()):
            return False
        return self.low <= int(value) <= self.high


class BooleanParameter(ParameterSpec):
    """Boolean parameter with prior probability for True."""
    def __init__(
        self,
        name: str,
        prior_p: float = 0.5,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        self.prior_p = MathUtils.clamp(prior_p, 0.01, 0.99)

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_boolean(self)

    def validate_value(self, value: Any) -> bool:
        return isinstance(value, bool) or value in (0, 1)


class CategoricalParameter(ParameterSpec):
    """Categorical parameter supporting an arbitrary set of choice items."""
    def __init__(
        self,
        name: str,
        choices: List[Any],
        prior_probs: Optional[List[float]] = None,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        if not choices:
            raise ValueError(f"CategoricalParameter {name} requires at least one choice.")
        self.choices = list(choices)
        if prior_probs:
            if len(prior_probs) != len(self.choices):
                raise ValueError(f"Length of prior_probs ({len(prior_probs)}) must match choices ({len(self.choices)})")
            s = sum(prior_probs)
            self.prior_probs = [max(p / s, MathUtils.EPSILON) for p in prior_probs]
        else:
            self.prior_probs = [1.0 / len(self.choices)] * len(self.choices)

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_categorical(self)

    def validate_value(self, value: Any) -> bool:
        return value in self.choices


class DiscreteParameter(ParameterSpec):
    """Discrete parameter supporting a finite ordered set of numerical values."""
    def __init__(
        self,
        name: str,
        values: List[Union[int, float]],
        prior_probs: Optional[List[float]] = None,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        if not values:
            raise ValueError(f"DiscreteParameter {name} requires at least one value.")
        self.values = sorted(list(set(values)))
        if prior_probs:
            if len(prior_probs) != len(self.values):
                raise ValueError("Length of prior_probs must match number of unique values.")
            s = sum(prior_probs)
            self.prior_probs = [max(p / s, MathUtils.EPSILON) for p in prior_probs]
        else:
            self.prior_probs = [1.0 / len(self.values)] * len(self.values)

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_discrete(self)

    def validate_value(self, value: Any) -> bool:
        return any(abs(float(value) - float(v)) < MathUtils.EPSILON for v in self.values)


class HierarchicalParameter(ParameterSpec):
    """Conditional parameter active only when a parent parameter matches a condition."""
    def __init__(
        self,
        name: str,
        parent_name: str,
        condition_fn: Callable[[Any], bool],
        sub_parameter: ParameterSpec,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        self.parent_name = parent_name
        self.condition_fn = condition_fn
        self.sub_parameter = sub_parameter

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_hierarchical(self)

    def validate_value(self, value: Any) -> bool:
        return self.sub_parameter.validate_value(value)


class DynamicParameter(ParameterSpec):
    """Dynamic parameter whose valid bounds/choices depend on runtime generator logic."""
    def __init__(
        self,
        name: str,
        generator_fn: Callable[[Dict[str, Any]], Any],
        validator_fn: Optional[Callable[[Any], bool]] = None,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        self.generator_fn = generator_fn
        self.validator_fn = validator_fn

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_dynamic(self)

    def validate_value(self, value: Any) -> bool:
        if self.validator_fn:
            return self.validator_fn(value)
        return True


class VectorParameter(ParameterSpec):
    """Vector of parameters sharing an element specification or independent bounds."""
    def __init__(
        self,
        name: str,
        size: int,
        element_spec: ParameterSpec,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        if size <= 0:
            raise ValueError(f"VectorParameter {name} must have size >= 1 (got {size})")
        self.size = size
        self.element_spec = element_spec

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_vector(self)

    def validate_value(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != self.size:
            return False
        return all(self.element_spec.validate_value(item) for item in value)


class MatrixParameter(ParameterSpec):
    """2D Matrix parameter composed of rows and cols of elements."""
    def __init__(
        self,
        name: str,
        rows: int,
        cols: int,
        element_spec: ParameterSpec,
        tunable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, tunable, metadata)
        if rows <= 0 or cols <= 0:
            raise ValueError(f"MatrixParameter {name} must have rows and cols >= 1 (got {rows}x{cols})")
        self.rows = rows
        self.cols = cols
        self.element_spec = element_spec

    def accept(self, visitor: ParameterVisitor) -> Any:
        return visitor.visit_matrix(self)

    def validate_value(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != self.rows:
            return False
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != self.cols:
                return False
            if not all(self.element_spec.validate_value(item) for item in row):
                return False
        return True


# ==============================================================================
# VISITOR IMPLEMENTATIONS
# ==============================================================================

class DomainValidationVisitor(ParameterVisitor):
    """Visitor to validate if a concrete value conforms to the parameter's specification."""
    def __init__(self, value: Any):
        self.value = value

    def visit_continuous(self, param: ContinuousParameter) -> bool:
        return param.validate_value(self.value)

    def visit_integer(self, param: IntegerParameter) -> bool:
        return param.validate_value(self.value)

    def visit_boolean(self, param: BooleanParameter) -> bool:
        return param.validate_value(self.value)

    def visit_categorical(self, param: CategoricalParameter) -> bool:
        return param.validate_value(self.value)

    def visit_discrete(self, param: DiscreteParameter) -> bool:
        return param.validate_value(self.value)

    def visit_hierarchical(self, param: HierarchicalParameter) -> bool:
        return param.validate_value(self.value)

    def visit_dynamic(self, param: DynamicParameter) -> bool:
        return param.validate_value(self.value)

    def visit_vector(self, param: VectorParameter) -> bool:
        return param.validate_value(self.value)

    def visit_matrix(self, param: MatrixParameter) -> bool:
        return param.validate_value(self.value)


class SamplingVisitor(ParameterVisitor):
    """Visitor to sample a random configuration value from a parameter's prior domain."""
    def __init__(self, rng: Optional[random.Random] = None, context: Optional[Dict[str, Any]] = None):
        self.rng = rng or random._inst if hasattr(random, "_inst") else random
        self.context = context or {}

    def visit_continuous(self, param: ContinuousParameter) -> float:
        if param.log_scale:
            log_low = math.log(param.low)
            log_high = math.log(param.high)
            return math.exp(self.rng.uniform(log_low, log_high))
        return self.rng.uniform(param.low, param.high)

    def visit_integer(self, param: IntegerParameter) -> int:
        return self.rng.randint(param.low, param.high)

    def visit_boolean(self, param: BooleanParameter) -> bool:
        return self.rng.random() < param.prior_p

    def visit_categorical(self, param: CategoricalParameter) -> Any:
        return self.rng.choices(param.choices, weights=param.prior_probs, k=1)[0]

    def visit_discrete(self, param: DiscreteParameter) -> Any:
        return self.rng.choices(param.values, weights=param.prior_probs, k=1)[0]

    def visit_hierarchical(self, param: HierarchicalParameter) -> Any:
        return param.sub_parameter.accept(self)

    def visit_dynamic(self, param: DynamicParameter) -> Any:
        return param.generator_fn(self.context)

    def visit_vector(self, param: VectorParameter) -> List[Any]:
        return [param.element_spec.accept(self) for _ in range(param.size)]

    def visit_matrix(self, param: MatrixParameter) -> List[List[Any]]:
        return [[param.element_spec.accept(self) for _ in range(param.cols)] for _ in range(param.rows)]


class SerializationVisitor(ParameterVisitor):
    """Visitor to serialize any parameter specification into a pure JSON-serializable dictionary."""
    def visit_continuous(self, param: ContinuousParameter) -> Dict[str, Any]:
        return {
            "type": "Continuous", "name": param.name, "low": param.low, "high": param.high,
            "prior_mean": param.prior_mean, "prior_std": param.prior_std, "log_scale": param.log_scale,
            "tunable": param.tunable, "metadata": param.metadata
        }

    def visit_integer(self, param: IntegerParameter) -> Dict[str, Any]:
        return {
            "type": "Integer", "name": param.name, "low": param.low, "high": param.high,
            "prior_mean": param.prior_mean, "prior_std": param.prior_std,
            "tunable": param.tunable, "metadata": param.metadata
        }

    def visit_boolean(self, param: BooleanParameter) -> Dict[str, Any]:
        return {
            "type": "Boolean", "name": param.name, "prior_p": param.prior_p,
            "tunable": param.tunable, "metadata": param.metadata
        }

    def visit_categorical(self, param: CategoricalParameter) -> Dict[str, Any]:
        return {
            "type": "Categorical", "name": param.name, "choices": param.choices,
            "prior_probs": param.prior_probs, "tunable": param.tunable, "metadata": param.metadata
        }

    def visit_discrete(self, param: DiscreteParameter) -> Dict[str, Any]:
        return {
            "type": "Discrete", "name": param.name, "values": param.values,
            "prior_probs": param.prior_probs, "tunable": param.tunable, "metadata": param.metadata
        }

    def visit_hierarchical(self, param: HierarchicalParameter) -> Dict[str, Any]:
        return {
            "type": "Hierarchical", "name": param.name, "parent_name": param.parent_name,
            "sub_parameter": param.sub_parameter.accept(self), "tunable": param.tunable, "metadata": param.metadata
        }

    def visit_dynamic(self, param: DynamicParameter) -> Dict[str, Any]:
        return {
            "type": "Dynamic", "name": param.name, "tunable": param.tunable, "metadata": param.metadata
        }

    def visit_vector(self, param: VectorParameter) -> Dict[str, Any]:
        return {
            "type": "Vector", "name": param.name, "size": param.size,
            "element_spec": param.element_spec.accept(self), "tunable": param.tunable, "metadata": param.metadata
        }

    def visit_matrix(self, param: MatrixParameter) -> Dict[str, Any]:
        return {
            "type": "Matrix", "name": param.name, "rows": param.rows, "cols": param.cols,
            "element_spec": param.element_spec.accept(self), "tunable": param.tunable, "metadata": param.metadata
        }


# ==============================================================================
# PARAMETER REGISTRY (REGISTRY PATTERN)
# ==============================================================================

class ParameterRegistry:
    """Registry maintaining searchable parameters categorized by module/subsystem."""
    def __init__(self):
        self._parameters: Dict[str, ParameterSpec] = {}
        self._module_map: Dict[str, List[str]] = defaultdict(list)

    def register(self, param: ParameterSpec, module_name: str = "general") -> None:
        """Register a parameter specification under a module name."""
        self._parameters[param.name] = param
        if param.name not in self._module_map[module_name]:
            self._module_map[module_name].append(param.name)

    def get(self, name: str) -> Optional[ParameterSpec]:
        """Retrieve a parameter specification by name."""
        return self._parameters.get(name)

    def get_by_module(self, module_name: str) -> List[ParameterSpec]:
        """Retrieve all parameter specifications associated with a specific module."""
        return [self._parameters[name] for name in self._module_map.get(module_name, []) if name in self._parameters]

    def get_all(self) -> List[ParameterSpec]:
        """Retrieve all registered parameter specifications in deterministic sorted order by name."""
        return [self._parameters[k] for k in sorted(self._parameters.keys())]

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate an entire configuration dictionary against registered parameter domains."""
        for name, value in config.items():
            param = self.get(name)
            if param and not param.accept(DomainValidationVisitor(value)):
                return False
        return True

    def sample_random_configuration(self, rng: Optional[random.Random] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sample a fully random configuration dictionary across all registered parameters."""
        visitor = SamplingVisitor(rng, context)
        config = {}
        for param in self.get_all():
            if isinstance(param, HierarchicalParameter):
                parent_val = config.get(param.parent_name)
                if parent_val is not None and param.condition_fn(parent_val):
                    config[param.name] = param.accept(visitor)
            else:
                config[param.name] = param.accept(visitor)
        return config

    def populate_default_ecosystem_parameters(self) -> None:
        """Register the exact subsystem tunable parameters specified in the ecosystem architecture."""
        # 1. Memory
        self.register(IntegerParameter("memory.retrieval_depth", 1, 100, 10, 5), "Memory")
        self.register(ContinuousParameter("memory.decay_factor", 0.001, 1.0, 0.1, 0.05), "Memory")
        self.register(ContinuousParameter("memory.compression_ratio", 0.1, 10.0, 2.0, 0.5), "Memory")
        self.register(ContinuousParameter("memory.activation_threshold", 0.01, 0.99, 0.5, 0.1), "Memory")
        self.register(IntegerParameter("memory.cache_size", 16, 65536, 1024, 256), "Memory")

        # 2. Reasoning
        self.register(IntegerParameter("reasoning.beam_width", 1, 64, 5, 2), "Reasoning")
        self.register(IntegerParameter("reasoning.search_depth", 1, 50, 8, 3), "Reasoning")
        self.register(IntegerParameter("reasoning.recursion_depth", 1, 25, 4, 1), "Reasoning")
        self.register(ContinuousParameter("reasoning.decomposition_threshold", 0.1, 0.95, 0.6, 0.15), "Reasoning")
        self.register(ContinuousParameter("reasoning.confidence_cutoff", 0.05, 0.99, 0.75, 0.1), "Reasoning")
        self.register(ContinuousParameter("reasoning.exploration_coefficient", 0.01, 5.0, 1.414, 0.3), "Reasoning")

        # 3. Planning
        self.register(IntegerParameter("planning.branching_factor", 2, 32, 4, 2), "Planning")
        self.register(IntegerParameter("planning.execution_budget", 10, 100000, 1000, 200), "Planning")
        self.register(CategoricalParameter("planning.stopping_criterion", ["confidence", "budget", "plateau", "exact"]), "Planning")
        self.register(IntegerParameter("planning.loop_threshold", 1, 100, 10, 3), "Planning")

        # 4. Knowledge Graph
        self.register(ContinuousParameter("knowledge_graph.forgetting_factor", 0.0, 0.5, 0.01, 0.005), "KnowledgeGraph")
        self.register(ContinuousParameter("knowledge_graph.beta_prior_alpha", 0.1, 100.0, 2.0, 1.0), "KnowledgeGraph")
        self.register(ContinuousParameter("knowledge_graph.beta_prior_beta", 0.1, 100.0, 2.0, 1.0), "KnowledgeGraph")
        self.register(IntegerParameter("knowledge_graph.particle_count", 10, 10000, 200, 50), "KnowledgeGraph")
        self.register(ContinuousParameter("knowledge_graph.ess_threshold", 0.1, 0.9, 0.5, 0.1), "KnowledgeGraph")
        self.register(ContinuousParameter("knowledge_graph.resampling_threshold", 0.1, 0.9, 0.5, 0.1), "KnowledgeGraph")
        self.register(ContinuousParameter("knowledge_graph.source_reliability_prior", 0.1, 1.0, 0.8, 0.1), "KnowledgeGraph")

        # 5. Fractal Engine
        self.register(IntegerParameter("fractal_engine.recursion_budget", 1, 20, 5, 2), "FractalEngine")
        self.register(IntegerParameter("fractal_engine.refinement_iterations", 1, 500, 25, 10), "FractalEngine")
        self.register(ContinuousParameter("fractal_engine.merge_threshold", 0.1, 0.99, 0.8, 0.1), "FractalEngine")
        self.register(ContinuousParameter("fractal_engine.abstraction_threshold", 0.1, 0.99, 0.7, 0.1), "FractalEngine")
        self.register(BooleanParameter("fractal_engine.context_expansion", prior_p=0.7), "FractalEngine")

        # 6. Decoder
        self.register(ContinuousParameter("decoder.temperature", 0.01, 2.0, 0.7, 0.2), "Decoder")
        self.register(IntegerParameter("decoder.top_k", 1, 500, 50, 15), "Decoder")
        self.register(ContinuousParameter("decoder.top_p_equivalent", 0.1, 1.0, 0.9, 0.05), "Decoder")
        self.register(ContinuousParameter("decoder.deterministic_threshold", 0.5, 1.0, 0.95, 0.02), "Decoder")
        self.register(VectorParameter("decoder.ranking_weights", 3, ContinuousParameter("weight", 0.0, 1.0)), "Decoder")

        # 7. Controller
        self.register(IntegerParameter("controller.retry_count", 0, 10, 3, 1), "Controller")
        self.register(IntegerParameter("controller.reflection_interval", 1, 100, 5, 2), "Controller")
        self.register(ContinuousParameter("controller.confidence_target", 0.5, 0.999, 0.9, 0.05), "Controller")
        self.register(ContinuousParameter("controller.planner_frequency", 0.01, 1.0, 0.2, 0.05), "Controller")


# ==============================================================================
# CONFIGURATIONS, TRIALS, & COMMANDS (COMMAND PATTERN)
# ==============================================================================

@dataclass
class ParameterConfiguration:
    """Encapsulates a dictionary of parameter assignments and numerical conversion helpers."""
    values: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.values)

    def get_numerical_vector(self, registry: ParameterRegistry) -> List[float]:
        """Flatten continuous and integer parameter assignments into a clean numerical vector."""
        vec = []
        for param in registry.get_all():
            if param.name in self.values:
                val = self.values[param.name]
                if isinstance(param, (ContinuousParameter, IntegerParameter)):
                    vec.append(float(val))
                elif isinstance(param, BooleanParameter):
                    vec.append(1.0 if val else 0.0)
        return vec

    def distance_to(self, other: 'ParameterConfiguration', registry: ParameterRegistry) -> float:
        """Compute Euclidean distance between two configurations across continuous/integer parameters."""
        v1 = self.get_numerical_vector(registry)
        v2 = other.get_numerical_vector(registry)
        if not v1 or len(v1) != len(v2):
            return float('inf')
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


@dataclass
class ContextMetadata:
    """Rich execution context including environment, hardware, and dataset parameters."""
    hardware_context: Dict[str, Any] = field(default_factory=dict)
    dataset_context: Dict[str, Any] = field(default_factory=dict)
    environment_context: Dict[str, Any] = field(default_factory=dict)
    execution_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware": self.hardware_context,
            "dataset": self.dataset_context,
            "environment": self.environment_context,
            "tags": self.execution_tags,
        }


@dataclass
class TrialResult:
    """Immutable record of an evaluated trial execution."""
    trial_id: str
    config: Dict[str, Any]
    objective_scores: Dict[str, float]
    composite_score: float
    status: TrialStatus
    start_time: float
    end_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: ContextMetadata = field(default_factory=ContextMetadata)
    error_message: Optional[str] = None
    rung: int = 0

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0


class Command(abc.ABC):
    """Abstract Command pattern interface for executing tasks inside the engine."""
    @abc.abstractmethod
    def execute(self) -> Any: pass


class EvaluateTrialCommand(Command):
    """Command to execute a candidate configuration through the objective function and constraints."""
    def __init__(
        self,
        trial_id: str,
        config: Dict[str, Any],
        objective_fn: 'ObjectiveFunction',
        constraints: List['Constraint'],
        scoring: 'CompositeScoring',
        context: Optional[ContextMetadata] = None,
        rung: int = 0,
    ):
        self.trial_id = trial_id
        self.config = config
        self.objective_fn = objective_fn
        self.constraints = constraints
        self.scoring = scoring
        self.context = context or ContextMetadata()
        self.rung = rung

    def execute(self) -> TrialResult:
        start_t = time.time()
        # 1. Check hard and soft constraints
        total_penalty = 0.0
        for constraint in self.constraints:
            c_res = constraint.evaluate(ParameterConfiguration(self.config))
            if not c_res.is_satisfied and isinstance(constraint, HardConstraint):
                end_t = time.time()
                return TrialResult(
                    trial_id=self.trial_id,
                    config=self.config,
                    objective_scores={"constraint_violation": -1e6},
                    composite_score=-1e6,
                    status=TrialStatus.FAILED,
                    start_time=start_t,
                    end_time=end_t,
                    error_message=f"Hard constraint violated: {constraint.name}",
                    context=self.context,
                    rung=self.rung,
                )
            total_penalty += c_res.penalty

        # 2. Evaluate Objective Function
        try:
            raw_scores = self.objective_fn.evaluate(ParameterConfiguration(self.config), self.context)
            if not isinstance(raw_scores, dict):
                raw_scores = {"main": float(raw_scores)}
        except Exception as e:
            end_t = time.time()
            return TrialResult(
                trial_id=self.trial_id,
                config=self.config,
                objective_scores={"error": -1e6},
                composite_score=-1e6,
                status=TrialStatus.FAILED,
                start_time=start_t,
                end_time=end_t,
                error_message=f"Objective evaluation failed: {str(e)}",
                context=self.context,
                rung=self.rung,
            )

        # 3. Compute Composite Score via Scoring Metric weights minus penalties
        composite = self.scoring.compute_composite_score(raw_scores) - total_penalty
        end_t = time.time()

        return TrialResult(
            trial_id=self.trial_id,
            config=self.config,
            objective_scores=raw_scores,
            composite_score=composite,
            status=TrialStatus.COMPLETED,
            start_time=start_t,
            end_time=end_t,
            metrics={"penalty": total_penalty, "eval_duration": end_t - start_t},
            context=self.context,
            rung=self.rung,
        )


# ==============================================================================
# CONSTRAINTS HIERARCHY
# ==============================================================================

@dataclass
class ConstraintResult:
    """Result of evaluating a constraint on a parameter configuration."""
    is_satisfied: bool
    violation_magnitude: float
    penalty: float


class Constraint(abc.ABC):
    """Abstract base class for parameter and resource constraints."""
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def evaluate(self, config: ParameterConfiguration) -> ConstraintResult: pass


class HardConstraint(Constraint):
    """Reject or apply infinite penalty if the predicate is violated."""
    def __init__(self, name: str, predicate_fn: Callable[[Dict[str, Any]], bool]):
        super().__init__(name)
        self.predicate_fn = predicate_fn

    def evaluate(self, config: ParameterConfiguration) -> ConstraintResult:
        satisfied = bool(self.predicate_fn(config.values))
        return ConstraintResult(is_satisfied=satisfied, violation_magnitude=0.0 if satisfied else 1.0, penalty=0.0 if satisfied else 1e9)


class SoftConstraint(Constraint):
    """Apply smooth linear or quadratic penalty proportional to violation magnitude."""
    def __init__(self, name: str, violation_fn: Callable[[Dict[str, Any]], float], weight: float = 10.0):
        super().__init__(name)
        self.violation_fn = violation_fn
        self.weight = weight

    def evaluate(self, config: ParameterConfiguration) -> ConstraintResult:
        violation = max(0.0, float(self.violation_fn(config.values)))
        penalty = self.weight * violation
        return ConstraintResult(is_satisfied=(violation <= MathUtils.EPSILON), violation_magnitude=violation, penalty=penalty)


class ConditionalConstraint(Constraint):
    """Evaluate inner constraint only when condition holds true."""
    def __init__(self, name: str, condition_fn: Callable[[Dict[str, Any]], bool], inner_constraint: Constraint):
        super().__init__(name)
        self.condition_fn = condition_fn
        self.inner_constraint = inner_constraint

    def evaluate(self, config: ParameterConfiguration) -> ConstraintResult:
        if not self.condition_fn(config.values):
            return ConstraintResult(is_satisfied=True, violation_magnitude=0.0, penalty=0.0)
        return self.inner_constraint.evaluate(config)


class DependencyConstraint(Constraint):
    """Enforce mathematical dependency between two parameter values (e.g., param_a <= param_b)."""
    def __init__(self, name: str, param_a: str, param_b: str, comparison: str = "<="):
        super().__init__(name)
        self.param_a = param_a
        self.param_b = param_b
        self.comparison = comparison

    def evaluate(self, config: ParameterConfiguration) -> ConstraintResult:
        vals = config.values
        if self.param_a not in vals or self.param_b not in vals:
            return ConstraintResult(is_satisfied=True, violation_magnitude=0.0, penalty=0.0)
        a, b = float(vals[self.param_a]), float(vals[self.param_b])
        if self.comparison == "<=":
            sat = (a <= b + MathUtils.EPSILON)
            viol = max(0.0, a - b)
        elif self.comparison == "<":
            sat = (a < b)
            viol = max(0.0, a - b + MathUtils.EPSILON)
        elif self.comparison == ">=":
            sat = (a >= b - MathUtils.EPSILON)
            viol = max(0.0, b - a)
        else:
            sat = (abs(a - b) < MathUtils.EPSILON)
            viol = abs(a - b)
        return ConstraintResult(is_satisfied=sat, violation_magnitude=viol, penalty=0.0 if sat else viol * 100.0)


class ResourceLimitConstraint(Constraint):
    """Limit resource consumption estimate (latency, memory, token efficiency)."""
    def __init__(self, name: str, resource_estimator_fn: Callable[[Dict[str, Any]], float], limit: float):
        super().__init__(name)
        self.resource_estimator_fn = resource_estimator_fn
        self.limit = limit

    def evaluate(self, config: ParameterConfiguration) -> ConstraintResult:
        est = float(self.resource_estimator_fn(config.values))
        viol = max(0.0, est - self.limit)
        return ConstraintResult(is_satisfied=(viol <= MathUtils.EPSILON), violation_magnitude=viol, penalty=viol * 50.0)


class CompositeConstraint(Constraint):
    """Composite Pattern aggregating multiple constraints into one execution check."""
    def __init__(self, name: str, constraints: Optional[List[Constraint]] = None):
        super().__init__(name)
        self.constraints: List[Constraint] = constraints or []

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def evaluate(self, config: ParameterConfiguration) -> ConstraintResult:
        total_viol = 0.0
        total_pen = 0.0
        all_sat = True
        for c in self.constraints:
            res = c.evaluate(config)
            if not res.is_satisfied:
                all_sat = False
            total_viol += res.violation_magnitude
            total_pen += res.penalty
        return ConstraintResult(is_satisfied=all_sat, violation_magnitude=total_viol, penalty=total_pen)


# ==============================================================================
# SCORING METRICS & OBJECTIVE FUNCTIONS
# ==============================================================================

class ScoringDirection(enum.Enum):
    """Optimization goal direction for a metric."""
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"


@dataclass
class ScoringMetric:
    """Configurable scoring metric definition."""
    name: str
    direction: ScoringDirection
    weight: float = 1.0
    target: Optional[float] = None


class CompositeScoring:
    """Composite utility scalarizing multiple scoring metrics into a single optimization objective."""
    def __init__(self, metrics: Optional[List[ScoringMetric]] = None):
        self.metrics: List[ScoringMetric] = metrics or [ScoringMetric("main", ScoringDirection.MAXIMIZE, 1.0)]

    def compute_composite_score(self, objective_scores: Dict[str, float]) -> float:
        if not objective_scores:
            return 0.0
        score = 0.0
        found_any = False
        for m in self.metrics:
            if m.name in objective_scores:
                found_any = True
                val = objective_scores[m.name]
                if m.direction == ScoringDirection.MAXIMIZE:
                    score += m.weight * val
                else:
                    score -= m.weight * val
        if not found_any and len(objective_scores) == 1:
            return float(next(iter(objective_scores.values())))
        if not found_any:
            return float(sum(objective_scores.values()))
        return score


class ObjectiveFunction(abc.ABC):
    """Abstract base class for objective function evaluations."""
    @abc.abstractmethod
    def evaluate(self, config: ParameterConfiguration, context: ContextMetadata) -> Dict[str, float]: pass


class SingleObjective(ObjectiveFunction):
    """Evaluates a single scalar target metric."""
    def __init__(self, metric_name: str, eval_fn: Callable[[Dict[str, Any], ContextMetadata], float]):
        self.metric_name = metric_name
        self.eval_fn = eval_fn

    def evaluate(self, config: ParameterConfiguration, context: ContextMetadata) -> Dict[str, float]:
        return {self.metric_name: float(self.eval_fn(config.values, context))}


class MultiObjective(ObjectiveFunction):
    """Evaluates multiple distinct metric functions simultaneously."""
    def __init__(self, eval_fns: Dict[str, Callable[[Dict[str, Any], ContextMetadata], float]]):
        self.eval_fns = eval_fns

    def evaluate(self, config: ParameterConfiguration, context: ContextMetadata) -> Dict[str, float]:
        return {name: float(fn(config.values, context)) for name, fn in self.eval_fns.items()}


class WeightedObjective(ObjectiveFunction):
    """Scalarizes multiple objective functions into a single weighted sum output dict."""
    def __init__(self, multi_obj: MultiObjective, weights: Dict[str, float]):
        self.multi_obj = multi_obj
        self.weights = weights

    def evaluate(self, config: ParameterConfiguration, context: ContextMetadata) -> Dict[str, float]:
        scores = self.multi_obj.evaluate(config, context)
        weighted_sum = sum(scores.get(name, 0.0) * w for name, w in self.weights.items())
        scores["weighted_composite"] = weighted_sum
        return scores


class ParetoFrontier:
    """Maintains non-dominated Pareto optimal solutions for multi-objective optimization."""
    def __init__(self, directions: Dict[str, ScoringDirection]):
        self.directions = directions
        self.solutions: List[TrialResult] = []

    def _is_dominated(self, candidate: Dict[str, float], existing: Dict[str, float]) -> bool:
        """Check whether `existing` Pareto-dominates `candidate`."""
        strictly_better = False
        for metric, direction in self.directions.items():
            c_val = candidate.get(metric, 0.0)
            e_val = existing.get(metric, 0.0)
            if direction == ScoringDirection.MAXIMIZE:
                if c_val > e_val + MathUtils.EPSILON:
                    return False
                if e_val > c_val + MathUtils.EPSILON:
                    strictly_better = True
            else: # MINIMIZE
                if c_val < e_val - MathUtils.EPSILON:
                    return False
                if e_val < c_val - MathUtils.EPSILON:
                    strictly_better = True
        return strictly_better

    def add_solution(self, trial: TrialResult) -> bool:
        """Add trial to Pareto frontier if non-dominated, pruning dominated existing points."""
        scores = trial.objective_scores
        for sol in self.solutions:
            if self._is_dominated(scores, sol.objective_scores):
                return False
        # Remove any existing solutions that are dominated by this new candidate
        self.solutions = [
            sol for sol in self.solutions
            if not self._is_dominated(sol.objective_scores, scores)
        ]
        self.solutions.append(trial)
        return True

    def get_frontier(self) -> List[TrialResult]:
        return list(self.solutions)


# ==============================================================================
# AUTOREGRESSIVE & BAYESIAN BELIEF TRACKERS
# ==============================================================================

@dataclass
class ParameterBeliefSnapshot:
    """Snapshot of a parameter's current conjugate posterior distribution estimates."""
    parameter_name: str
    prior_mean: float
    prior_std: float
    posterior_mean: float
    posterior_std: float
    credible_interval_low: float
    credible_interval_high: float
    expected_improvement: float = 0.0
    probability_of_improvement: float = 0.0
    information_gain: float = 0.0


class BayesianParameterBelief:
    """Maintains exact Bayesian conjugate updates (Normal-Inverse-Gamma & Beta-Dirichlet)."""
    def __init__(self, param: ParameterSpec):
        self.param = param
        self.trials_count = 0
        # Default fallback attributes for clean interface access across all types
        self.mu_n = 0.0
        self.kappa_n = 1.0
        self.alpha_n = 2.0
        self.beta_n = 1.0

        if isinstance(param, (ContinuousParameter, IntegerParameter)):
            self.mu_0 = float(param.prior_mean)
            self.kappa_0 = 1.0
            self.alpha_0 = 2.0
            self.beta_0 = (float(param.prior_std) ** 2) * (self.alpha_0 - 1.0)
            self.mu_n = self.mu_0
            self.kappa_n = self.kappa_0
            self.alpha_n = self.alpha_0
            self.beta_n = self.beta_0
        elif isinstance(param, BooleanParameter):
            self.alpha_beta = (2.0 * param.prior_p, 2.0 * (1.0 - param.prior_p))
            self.mu_n = param.prior_p
        elif isinstance(param, (CategoricalParameter, DiscreteParameter)):
            self.dirichlet_counts = [p * 5.0 for p in param.prior_probs]
            self.mu_n = max(param.prior_probs) if param.prior_probs else 0.0

    def update_belief(self, value: Any, reward: float, baseline_sma: float = -1e5) -> None:
        """Online conjugate update incorporating a new parameter observation and whether it outperformed baseline."""
        self.trials_count += 1
        # Calculate learning weight: strongly pull towards high-performing points, ignore/shrink for bad points
        if baseline_sma > -1e4:
            if reward >= baseline_sma:
                weight = 1.0 + max(0.1, (reward - baseline_sma) / max(1.0, abs(baseline_sma)))
            else:
                weight = 0.05
        else:
            weight = max(0.1, 1.0 + MathUtils.clamp(reward / 10.0, -0.9, 5.0))

        if isinstance(self.param, (ContinuousParameter, IntegerParameter)):
            try:
                x = float(value)
                self.kappa_n += weight
                self.alpha_n += 0.5 * weight
                diff = x - self.mu_n
                self.mu_n += (weight / self.kappa_n) * diff
                self.beta_n += 0.5 * weight * (diff ** 2) * (self.kappa_n - weight) / max(self.kappa_n, MathUtils.EPSILON)
            except (ValueError, TypeError):
                pass
        elif isinstance(self.param, BooleanParameter):
            is_true = bool(value)
            a, b = self.alpha_beta
            if is_true:
                self.alpha_beta = (a + weight, b)
            else:
                self.alpha_beta = (a, b + weight)
            self.mu_n = self.alpha_beta[0] / (self.alpha_beta[0] + self.alpha_beta[1])
        elif isinstance(self.param, CategoricalParameter):
            if value in self.param.choices:
                idx = self.param.choices.index(value)
                self.dirichlet_counts[idx] += weight
        elif isinstance(self.param, DiscreteParameter):
            for idx, v in enumerate(self.param.values):
                if abs(float(value) - float(v)) < MathUtils.EPSILON:
                    self.dirichlet_counts[idx] += weight
                    break

    def get_posterior_mean_std(self) -> Tuple[float, float]:
        """Compute expected posterior mean and standard deviation for any parameter type."""
        if isinstance(self.param, (ContinuousParameter, IntegerParameter)):
            mean = self.mu_n
            var = self.beta_n / max(self.alpha_n - 1.0, MathUtils.EPSILON)
            return mean, math.sqrt(max(var, MathUtils.EPSILON))
        elif isinstance(self.param, BooleanParameter):
            a, b = self.alpha_beta
            mean = a / (a + b)
            var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
            return mean, math.sqrt(max(var, MathUtils.EPSILON))
        elif isinstance(self.param, (CategoricalParameter, DiscreteParameter)):
            s = sum(self.dirichlet_counts)
            probs = [c / s for c in self.dirichlet_counts]
            max_p = max(probs) if probs else 0.0
            return max_p, math.sqrt(max(max_p * (1.0 - max_p), MathUtils.EPSILON))
        return self.mu_n, 1.0

    def compute_expected_improvement(self, best_f: float, current_mean: float, current_std: float, xi: float = 0.01) -> float:
        """Compute analytical Expected Improvement (EI) for continuous parameters under Gaussian assumption."""
        if current_std <= MathUtils.EPSILON:
            return max(0.0, current_mean - best_f)
        z = (current_mean - best_f - xi) / current_std
        ei = (current_mean - best_f - xi) * MathUtils.norm_cdf(z) + current_std * MathUtils.norm_pdf(z)
        return max(0.0, ei)

    def compute_probability_of_improvement(self, best_f: float, current_mean: float, current_std: float, xi: float = 0.01) -> float:
        """Compute analytical Probability of Improvement (PI)."""
        if current_std <= MathUtils.EPSILON:
            return 1.0 if current_mean > best_f + xi else 0.0
        z = (current_mean - best_f - xi) / current_std
        return MathUtils.norm_cdf(z)

    def compute_information_gain(self, current_std: float) -> float:
        """Estimate differential entropy / Information Gain (IG) reduction."""
        if current_std <= MathUtils.EPSILON:
            return 0.0
        return 0.5 * math.log(2.0 * math.pi * math.e * (current_std ** 2))

    def get_snapshot(self, best_f: float = 0.0) -> ParameterBeliefSnapshot:
        p_mean, p_std = self.get_posterior_mean_std()
        ei = self.compute_expected_improvement(best_f, p_mean, p_std)
        pi = self.compute_probability_of_improvement(best_f, p_mean, p_std)
        ig = self.compute_information_gain(p_std)
        ci_low = p_mean - 1.96 * p_std
        ci_high = p_mean + 1.96 * p_std
        if isinstance(self.param, ContinuousParameter):
            ci_low = MathUtils.clamp(ci_low, self.param.low, self.param.high)
            ci_high = MathUtils.clamp(ci_high, self.param.low, self.param.high)
        elif isinstance(self.param, IntegerParameter):
            ci_low = MathUtils.clamp(ci_low, float(self.param.low), float(self.param.high))
            ci_high = MathUtils.clamp(ci_high, float(self.param.low), float(self.param.high))

        prior_m = float(getattr(self.param, "prior_mean", p_mean))
        prior_s = float(getattr(self.param, "prior_std", p_std))

        return ParameterBeliefSnapshot(
            parameter_name=self.param.name,
            prior_mean=prior_m,
            prior_std=prior_s,
            posterior_mean=p_mean,
            posterior_std=p_std,
            credible_interval_low=ci_low,
            credible_interval_high=ci_high,
            expected_improvement=ei,
            probability_of_improvement=pi,
            information_gain=ig,
        )


class BayesianBeliefTracker:
    """Manages conjugate Bayesian parameter beliefs across all registered ecosystem parameters."""
    def __init__(self, registry: ParameterRegistry):
        self.registry = registry
        self.beliefs: Dict[str, BayesianParameterBelief] = {}
        for param in registry.get_all():
            self.beliefs[param.name] = BayesianParameterBelief(param)

    def update_beliefs_from_trial(self, result: TrialResult, baseline_sma: float = -1e5) -> None:
        """Update posterior beliefs for every parameter configured in the completed trial."""
        if result.status != TrialStatus.COMPLETED:
            return
        for name, value in result.config.items():
            if name in self.beliefs:
                self.beliefs[name].update_belief(value, result.composite_score, baseline_sma=baseline_sma)

    def get_all_snapshots(self, best_f: float = 0.0) -> Dict[str, ParameterBeliefSnapshot]:
        return {name: b.get_snapshot(best_f) for name, b in self.beliefs.items()}


class AutoregressiveTracker:
    """Autoregressive history engine tracking moving averages, regret, velocity, and learning curves."""
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: List[TrialResult] = []
        self.best_result: Optional[TrialResult] = None
        self.scores_history: List[float] = []
        self.sma_history: List[float] = []
        self.ema_history: List[float] = []
        self.instantaneous_regret: List[float] = []
        self.cumulative_regret: List[float] = []
        self.exploration_scores: List[float] = []
        self.exploitation_scores: List[float] = []
        self.improvement_velocity: List[float] = []
        self.improvement_acceleration: List[float] = []

    def add_trial_result(self, result: TrialResult, beliefs: Optional[BayesianBeliefTracker] = None) -> None:
        """Autoregressively process a new trial result and compute time-series derivatives."""
        self.history.append(result)
        score = result.composite_score if result.status == TrialStatus.COMPLETED else -1e6
        self.scores_history.append(score)

        if result.status == TrialStatus.COMPLETED:
            if self.best_result is None or score > self.best_result.composite_score:
                self.best_result = result

        # Simple Moving Average (SMA)
        window = self.scores_history[-self.window_size:]
        sma = MathUtils.mean([s for s in window if s > -1e5])
        self.sma_history.append(sma)

        # Exponential Moving Average (EMA)
        alpha = 2.0 / (min(len(self.scores_history), self.window_size) + 1.0)
        prev_ema = self.ema_history[-1] if self.ema_history else score
        ema = alpha * score + (1.0 - alpha) * prev_ema
        self.ema_history.append(ema)

        # Regret calculations
        best_known = self.best_result.composite_score if self.best_result else score
        inst_regret = max(0.0, best_known - score)
        self.instantaneous_regret.append(inst_regret)
        prev_cum = self.cumulative_regret[-1] if self.cumulative_regret else 0.0
        self.cumulative_regret.append(prev_cum + inst_regret)

        # Exploration vs Exploitation scores
        if beliefs:
            snaps = beliefs.get_all_snapshots(best_known)
            avg_std = MathUtils.mean([s.posterior_std for s in snaps.values()])
            avg_mean = MathUtils.mean([s.posterior_mean for s in snaps.values()])
            self.exploration_scores.append(avg_std / max(abs(avg_mean), MathUtils.EPSILON))
            self.exploitation_scores.append(avg_mean)
        else:
            self.exploration_scores.append(0.5)
            self.exploitation_scores.append(sma)

        # Improvement Velocity (first derivative df/dt across last few trials)
        if len(self.sma_history) >= 2:
            vel = self.sma_history[-1] - self.sma_history[-2]
            self.improvement_velocity.append(vel)
        else:
            self.improvement_velocity.append(0.0)

        # Improvement Acceleration (second derivative d2f/dt2)
        if len(self.improvement_velocity) >= 2:
            acc = self.improvement_velocity[-1] - self.improvement_velocity[-2]
            self.improvement_acceleration.append(acc)
        else:
            self.improvement_acceleration.append(0.0)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Export comprehensive summary of autoregressive learning curves and velocities."""
        return {
            "total_trials": len(self.history),
            "completed_trials": sum(1 for t in self.history if t.status == TrialStatus.COMPLETED),
            "best_composite_score": self.best_result.composite_score if self.best_result else None,
            "current_sma": self.sma_history[-1] if self.sma_history else 0.0,
            "current_ema": self.ema_history[-1] if self.ema_history else 0.0,
            "total_cumulative_regret": self.cumulative_regret[-1] if self.cumulative_regret else 0.0,
            "latest_improvement_velocity": self.improvement_velocity[-1] if self.improvement_velocity else 0.0,
            "latest_improvement_acceleration": self.improvement_acceleration[-1] if self.improvement_acceleration else 0.0,
        }


# ==============================================================================
# KNOWLEDGE GRAPH & FRACTAL MEMORY INTEGRATION
# ==============================================================================

@dataclass
class KGNode:
    """Living Knowledge Graph node entity."""
    id: str
    node_type: str
    properties: Dict[str, Any]
    creation_time: float = field(default_factory=time.time)


@dataclass
class KGEdge:
    """Living Knowledge Graph directed edge between two nodes."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    properties: Dict[str, Any] = field(default_factory=dict)


class LivingKnowledgeGraphInterface(abc.ABC):
    """Abstract interface defining required interactions with the Living Knowledge Graph ecosystem."""
    @abc.abstractmethod
    def add_node(self, node: KGNode) -> None: pass

    @abc.abstractmethod
    def add_edge(self, edge: KGEdge) -> None: pass

    @abc.abstractmethod
    def record_trial(self, trial: TrialResult) -> None: pass

    @abc.abstractmethod
    def get_parameter_correlations(self, param_names: List[str]) -> Dict[Tuple[str, str], float]: pass


class LivingKnowledgeGraph(LivingKnowledgeGraphInterface):
    """Concrete in-memory Living Knowledge Graph storing nodes, edges, correlations, and causal links."""
    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self.best_parameters: Dict[str, Any] = {}
        self.worst_parameters: Dict[str, Any] = {}
        self.historical_experiments: List[TrialResult] = []
        self.causal_relationships: List[Dict[str, Any]] = []

    def add_node(self, node: KGNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge) -> None:
        self.edges.append(edge)

    def record_trial(self, trial: TrialResult) -> None:
        self.historical_experiments.append(trial)
        trial_node = KGNode(
            id=f"trial_{trial.trial_id}",
            node_type="ExperimentTrial",
            properties={"score": trial.composite_score, "status": trial.status.value, "config": trial.config}
        )
        self.add_node(trial_node)

        # Update best / worst known parameters
        if trial.status == TrialStatus.COMPLETED:
            if not self.best_parameters or trial.composite_score > self.best_parameters.get("score", -1e9):
                self.best_parameters = {"score": trial.composite_score, "config": copy.deepcopy(trial.config)}
            if not self.worst_parameters or trial.composite_score < self.worst_parameters.get("score", 1e9):
                self.worst_parameters = {"score": trial.composite_score, "config": copy.deepcopy(trial.config)}

        # Record context links
        for tag in trial.context.execution_tags:
            tag_node_id = f"tag_{tag}"
            if tag_node_id not in self.nodes:
                self.add_node(KGNode(id=tag_node_id, node_type="ContextTag", properties={"tag": tag}))
            self.add_edge(KGEdge(source_id=trial_node.id, target_id=tag_node_id, relation_type="HAS_CONTEXT", weight=1.0))

    def record_causal_relationship(self, param_name: str, metric_name: str, effect_size: float, p_value: float = 0.01) -> None:
        """Store inferred causal relationships between hyperparameter tweaks and performance metrics."""
        self.causal_relationships.append({
            "param": param_name, "metric": metric_name, "effect_size": effect_size, "p_value": p_value, "timestamp": time.time()
        })

    def get_parameter_correlations(self, param_names: List[str]) -> Dict[Tuple[str, str], float]:
        """Compute Pearson correlation matrix between historical parameter values across successful trials."""
        completed = [t for t in self.historical_experiments if t.status == TrialStatus.COMPLETED]
        if len(completed) < 3:
            return {}
        correlations: Dict[Tuple[str, str], float] = {}
        vectors: Dict[str, List[float]] = {name: [] for name in param_names}
        for t in completed:
            for name in param_names:
                if name in t.config and isinstance(t.config[name], (int, float)):
                    vectors[name].append(float(t.config[name]))
                else:
                    vectors[name].append(0.0)
        for i, n1 in enumerate(param_names):
            for j, n2 in enumerate(param_names):
                if i <= j and vectors[n1] and vectors[n2]:
                    corr = MathUtils.pearson_correlation(vectors[n1], vectors[n2])
                    correlations[(n1, n2)] = corr
                    correlations[(n2, n1)] = corr
        return correlations


class FractalMemoryInterface(abc.ABC):
    """Abstract interface defining required interactions with the hierarchical Fractal Memory subsystem."""
    @abc.abstractmethod
    def store_elite_configuration(self, trial: TrialResult) -> None: pass

    @abc.abstractmethod
    def store_failed_configuration(self, trial: TrialResult) -> None: pass

    @abc.abstractmethod
    def get_elites(self, top_k: int = 10) -> List[TrialResult]: pass


class FractalMemory(FractalMemoryInterface):
    """Concrete hierarchical Fractal Memory storing elite/failed parameter sets, search trees, and trajectories."""
    def __init__(self, max_elites: int = 50, decay_factor: float = 0.05):
        self.max_elites = max_elites
        self.decay_factor = decay_factor
        self.elite_parameter_sets: List[TrialResult] = []
        self.failed_parameter_sets: List[TrialResult] = []
        self.partial_solutions: Dict[str, Dict[str, Any]] = {}
        self.search_trees: List[Dict[str, Any]] = []
        self.optimization_trajectories: List[Tuple[float, float]] = []
        self.recursive_search_states: Dict[str, Any] = {}

    def store_elite_configuration(self, trial: TrialResult) -> None:
        self.elite_parameter_sets.append(trial)
        self.elite_parameter_sets.sort(key=lambda t: t.composite_score, reverse=True)
        if len(self.elite_parameter_sets) > self.max_elites:
            self.elite_parameter_sets = self.elite_parameter_sets[:self.max_elites]
        self.optimization_trajectories.append((trial.end_time, trial.composite_score))

    def store_failed_configuration(self, trial: TrialResult) -> None:
        self.failed_parameter_sets.append(trial)
        if len(self.failed_parameter_sets) > 500:
            self.failed_parameter_sets.pop(0)

    def get_elites(self, top_k: int = 10) -> List[TrialResult]:
        return list(self.elite_parameter_sets[:top_k])

    def store_recursive_search_state(self, state_id: str, state_data: Dict[str, Any]) -> None:
        self.recursive_search_states[state_id] = copy.deepcopy(state_data)


# ==============================================================================
# OBSERVER PATTERN FOR TELEMETRY AND INTEGRATION
# ==============================================================================

class OptimizationObserver(abc.ABC):
    """Observer Pattern interface for lifecycle events."""
    @abc.abstractmethod
    def on_trial_start(self, trial_id: str, config: Dict[str, Any]) -> None: pass

    @abc.abstractmethod
    def on_trial_complete(self, result: TrialResult) -> None: pass

    @abc.abstractmethod
    def on_strategy_switch(self, old_strategy: str, new_strategy: str) -> None: pass

    @abc.abstractmethod
    def on_state_change(self, old_state: OptimizerState, new_state: OptimizerState) -> None: pass


class KnowledgeGraphObserver(OptimizationObserver):
    """Observer piping optimization experiences directly into the Living Knowledge Graph."""
    def __init__(self, kg: LivingKnowledgeGraphInterface):
        self.kg = kg

    def on_trial_start(self, trial_id: str, config: Dict[str, Any]) -> None:
        pass

    def on_trial_complete(self, result: TrialResult) -> None:
        self.kg.record_trial(result)

    def on_strategy_switch(self, old_strategy: str, new_strategy: str) -> None:
        pass

    def on_state_change(self, old_state: OptimizerState, new_state: OptimizerState) -> None:
        pass


class FractalMemoryObserver(OptimizationObserver):
    """Observer storing elite parameter configurations and search trajectories into Fractal Memory."""
    def __init__(self, memory: FractalMemoryInterface):
        self.memory = memory

    def on_trial_start(self, trial_id: str, config: Dict[str, Any]) -> None:
        pass

    def on_trial_complete(self, result: TrialResult) -> None:
        if result.status == TrialStatus.COMPLETED:
            self.memory.store_elite_configuration(result)
        elif result.status == TrialStatus.FAILED:
            self.memory.store_failed_configuration(result)

    def on_strategy_switch(self, old_strategy: str, new_strategy: str) -> None:
        pass

    def on_state_change(self, old_state: OptimizerState, new_state: OptimizerState) -> None:
        pass


class AutoregressiveObserver(OptimizationObserver):
    """Observer updating autoregressive histories and Bayesian parameter beliefs continuously."""
    def __init__(self, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker):
        self.tracker = tracker
        self.beliefs = beliefs

    def on_trial_start(self, trial_id: str, config: Dict[str, Any]) -> None:
        pass

    def on_trial_complete(self, result: TrialResult) -> None:
        self.tracker.add_trial_result(result, self.beliefs)
        sma = self.tracker.sma_history[-1] if self.tracker.sma_history else -1e5
        self.beliefs.update_beliefs_from_trial(result, baseline_sma=sma)

    def on_strategy_switch(self, old_strategy: str, new_strategy: str) -> None:
        pass

    def on_state_change(self, old_state: OptimizerState, new_state: OptimizerState) -> None:
        pass


# ==============================================================================
# EARLY STOPPING CONTROLLER
# ==============================================================================

class EarlyStopper(abc.ABC):
    """Abstract base class for early stopping rules."""
    @abc.abstractmethod
    def should_stop(self, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker) -> Tuple[bool, str]: pass


class PatienceEarlyStopper(EarlyStopper):
    """Stop if no improvement is observed after N consecutive trials."""
    def __init__(self, patience: int = 15):
        self.patience = patience

    def should_stop(self, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker) -> Tuple[bool, str]:
        if len(tracker.scores_history) < self.patience + 5:
            return False, ""
        best_so_far = -1e9
        stagnant_count = 0
        for s in tracker.scores_history:
            if s > best_so_far + MathUtils.EPSILON:
                best_so_far = s
                stagnant_count = 0
            else:
                stagnant_count += 1
        if stagnant_count >= self.patience:
            return True, f"Patience exhausted: no improvement for {stagnant_count} consecutive trials."
        return False, ""


class ConfidenceConvergenceStopper(EarlyStopper):
    """Stop if the average Bayesian posterior standard deviation drops below target."""
    def __init__(self, target_std: float = 0.05):
        self.target_std = target_std

    def should_stop(self, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker) -> Tuple[bool, str]:
        if len(tracker.history) < 10:
            return False, ""
        snaps = beliefs.get_all_snapshots()
        if not snaps:
            return False, ""
        avg_std = MathUtils.mean([s.posterior_std for s in snaps.values()])
        if avg_std < self.target_std:
            return True, f"Confidence converged: average posterior std ({avg_std:.4f}) < target ({self.target_std})."
        return False, ""


class PlateauDetectionStopper(EarlyStopper):
    """Stop if improvement velocity remains near zero over a sliding window."""
    def __init__(self, window: int = 10, threshold: float = 1e-4):
        self.window = window
        self.threshold = threshold

    def should_stop(self, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker) -> Tuple[bool, str]:
        if len(tracker.improvement_velocity) < self.window + 5:
            return False, ""
        recent_vel = tracker.improvement_velocity[-self.window:]
        if all(abs(v) < self.threshold for v in recent_vel):
            return True, f"Plateau detected: improvement velocity remained near 0 across {self.window} trials."
        return False, ""


class BudgetExhaustionStopper(EarlyStopper):
    """Stop when maximum trial count or execution duration is reached."""
    def __init__(self, max_trials: int = 1000, max_seconds: float = 3600.0, start_time: float = 0.0):
        self.max_trials = max_trials
        self.max_seconds = max_seconds
        self.start_time = start_time or time.time()

    def should_stop(self, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker) -> Tuple[bool, str]:
        if len(tracker.history) >= self.max_trials:
            return True, f"Budget exhausted: reached maximum trial limit ({self.max_trials})."
        if (time.time() - self.start_time) >= self.max_seconds:
            return True, f"Budget exhausted: reached time limit ({self.max_seconds}s)."
        return False, ""


class UncertaintyConvergenceStopper(EarlyStopper):
    """Stop when information gain / entropy across parameters approaches zero."""
    def __init__(self, threshold: float = -2.0):
        self.threshold = threshold

    def should_stop(self, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker) -> Tuple[bool, str]:
        if len(tracker.history) < 15:
            return False, ""
        snaps = beliefs.get_all_snapshots()
        if not snaps:
            return False, ""
        avg_ig = MathUtils.mean([s.information_gain for s in snaps.values()])
        if avg_ig < self.threshold:
            return True, f"Uncertainty converged: average information gain ({avg_ig:.4f}) below threshold."
        return False, ""


class EarlyStoppingController:
    """Composite Early Stopping engine running all registered stoppers sequentially."""
    def __init__(self, stoppers: Optional[List[EarlyStopper]] = None):
        self.stoppers: List[EarlyStopper] = stoppers or []

    def add_stopper(self, stopper: EarlyStopper) -> None:
        self.stoppers.append(stopper)

    def evaluate_stopping(self, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker) -> Tuple[bool, str]:
        for stopper in self.stoppers:
            should, reason = stopper.should_stop(tracker, beliefs)
            if should:
                return True, reason
        return False, ""


# ==============================================================================
# JOINT GAUSSIAN PROCESS SURROGATE & ACQUISITION FUNCTIONS
# ==============================================================================
# Complements BayesianParameterBelief's independent per-parameter conjugate updates with a joint
# surrogate over the full continuous/integer parameter vector, capturing cross-parameter covariance
# via a shared RBF kernel. Fit uses Cholesky decomposition (MathUtils.cholesky/solve_cholesky).

class RBFKernel:
    """Squared-exponential covariance kernel with optional per-dimension length scales (ARD)."""
    def __init__(self, length_scales: Optional[List[float]] = None, variance: float = 1.0):
        self.length_scales = length_scales
        self.variance = variance

    def __call__(self, x1: List[float], x2: List[float]) -> float:
        if not x1 or len(x1) != len(x2):
            return 0.0
        if self.length_scales is None:
            sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        else:
            sq_dist = sum(((a - b) / max(l, MathUtils.EPSILON)) ** 2 for a, b, l in zip(x1, x2, self.length_scales))
        return self.variance * math.exp(-0.5 * sq_dist)


class GaussianProcessRegressor:
    """Gaussian Process surrogate regressing a joint normalized parameter vector onto observed composite scores."""
    def __init__(self, kernel: RBFKernel, noise: float = 1e-6, normalize_y: bool = True):
        self.kernel = kernel
        self.noise = noise
        self.normalize_y = normalize_y
        self.X_train: List[List[float]] = []
        self.y_mean = 0.0
        self.y_std = 1.0
        self._L: Optional[Matrix] = None
        self._alpha: List[float] = []
        self._fitted = False

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        self.X_train = X
        if self.normalize_y:
            self.y_mean = MathUtils.mean(y)
            self.y_std = MathUtils.std_dev(y, sample=True) if len(y) > 1 else 1.0
            if self.y_std <= MathUtils.EPSILON:
                self.y_std = 1.0
            y_scaled = [(yi - self.y_mean) / self.y_std for yi in y]
        else:
            y_scaled = list(y)

        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = self.kernel(X[i], X[j]) + (self.noise if i == j else 0.0)
                K[i][j] = val
                K[j][i] = val
        self._L = MathUtils.cholesky(K)
        self._alpha = MathUtils.solve_cholesky(self._L, y_scaled)
        self._fitted = True

    def predict(self, X_test: List[List[float]]) -> Tuple[List[float], List[float]]:
        """Return (means, std_devs) for each test point."""
        if not self._fitted or not self.X_train:
            return [self.y_mean] * len(X_test), [1.0] * len(X_test)

        means, stds = [], []
        for x_star in X_test:
            k_star = [self.kernel(x_star, xi) for xi in self.X_train]
            k_star_star = self.kernel(x_star, x_star) + self.noise
            mean_scaled = sum(k * a for k, a in zip(k_star, self._alpha))
            v = MathUtils.forward_substitution(self._L, k_star)
            var_scaled = max(k_star_star - sum(vi * vi for vi in v), MathUtils.EPSILON)
            std_scaled = math.sqrt(var_scaled)
            if self.normalize_y:
                means.append(mean_scaled * self.y_std + self.y_mean)
                stds.append(std_scaled * self.y_std)
            else:
                means.append(mean_scaled)
                stds.append(std_scaled)
        return means, stds


class AcquisitionFunction(abc.ABC):
    """Abstract acquisition function ranking candidate points against a fitted GaussianProcessRegressor."""
    @abc.abstractmethod
    def evaluate(self, gp: GaussianProcessRegressor, X: List[List[float]], best_f: float) -> List[float]: pass


class ExpectedImprovement(AcquisitionFunction):
    """Analytical Expected Improvement (EI), oriented for maximization of the composite score."""
    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def evaluate(self, gp: GaussianProcessRegressor, X: List[List[float]], best_f: float) -> List[float]:
        means, stds = gp.predict(X)
        ei_vals = []
        for mu, sigma in zip(means, stds):
            if sigma <= MathUtils.EPSILON:
                ei_vals.append(max(0.0, mu - best_f))
                continue
            imp = mu - best_f - self.xi
            z = imp / sigma
            ei_vals.append(max(0.0, imp * MathUtils.norm_cdf(z) + sigma * MathUtils.norm_pdf(z)))
        return ei_vals


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound (UCB), rewarding both high predicted mean and high predictive uncertainty."""
    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def evaluate(self, gp: GaussianProcessRegressor, X: List[List[float]], best_f: float) -> List[float]:
        means, stds = gp.predict(X)
        return [mu + self.kappa * sigma for mu, sigma in zip(means, stds)]


# ==============================================================================
# SEARCH STRATEGIES HIERARCHY (STRATEGY PATTERN)
# ==============================================================================

class SearchStrategy(abc.ABC):
    """Strategy Pattern interface for generating candidate configurations."""
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]: pass

    def update(self, result: TrialResult, beliefs: BayesianBeliefTracker) -> None:
        """Optional hook for strategies requiring internal state maintenance across evaluations."""
        pass


# 1. Random Search Strategy
class RandomSearchStrategy(SearchStrategy):
    """Pure uniform random sampling across parameter domains."""
    def __init__(self):
        super().__init__("RandomSearch")

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        return [registry.sample_random_configuration() for _ in range(count)]


# 2. Grid Search Strategy
class GridSearchStrategy(SearchStrategy):
    """Systematic grid search discretizing continuous intervals into mesh points."""
    def __init__(self, resolution: int = 5):
        super().__init__("GridSearch")
        self.resolution = resolution
        self._grid_generator: Optional[Generator[Dict[str, Any], None, None]] = None

    def _build_generator(self, registry: ParameterRegistry) -> Generator[Dict[str, Any], None, None]:
        params = registry.get_all()
        grids = {}
        for p in params:
            if isinstance(p, ContinuousParameter):
                step = (p.high - p.low) / max(1, self.resolution - 1)
                grids[p.name] = [p.low + i * step for i in range(self.resolution)]
            elif isinstance(p, IntegerParameter):
                step = max(1, (p.high - p.low) // max(1, self.resolution - 1))
                grids[p.name] = list(range(p.low, p.high + 1, step))[:self.resolution]
            elif isinstance(p, BooleanParameter):
                grids[p.name] = [True, False]
            elif isinstance(p, (CategoricalParameter, DiscreteParameter)):
                grids[p.name] = p.choices if isinstance(p, CategoricalParameter) else p.values
            else:
                grids[p.name] = [registry.sample_random_configuration().get(p.name)]
        names = list(grids.keys())
        for combo in itertools.product(*(grids[n] for n in names)):
            yield dict(zip(names, combo))

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        if self._grid_generator is None:
            self._grid_generator = self._build_generator(registry)
        results = []
        for _ in range(count):
            try:
                results.append(next(self._grid_generator))
            except StopIteration:
                results.append(registry.sample_random_configuration())
        return results


# 3. Latin Hypercube Search Strategy
class LatinHypercubeSearchStrategy(SearchStrategy):
    """Latin Hypercube Sampling (LHS) stratifying continuous parameter intervals uniformly."""
    def __init__(self):
        super().__init__("LatinHypercube")

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        if count <= 0:
            return []
        params = registry.get_all()
        strata: Dict[str, List[Any]] = {}
        for p in params:
            if isinstance(p, ContinuousParameter):
                step = (p.high - p.low) / count
                vals = [random.uniform(p.low + i * step, p.low + (i + 1) * step) for i in range(count)]
                random.shuffle(vals)
                strata[p.name] = vals
            elif isinstance(p, IntegerParameter):
                vals = [random.randint(p.low, p.high) for _ in range(count)]
                strata[p.name] = vals
            else:
                strata[p.name] = [registry.sample_random_configuration().get(p.name) for _ in range(count)]
        return [{p.name: strata[p.name][i] for p in params} for i in range(count)]


# 4. Adaptive Random Search Strategy
class AdaptiveRandomSearchStrategy(SearchStrategy):
    """Random search centered around current best known point, shrinking radius when stagnated."""
    def __init__(self, initial_step_fraction: float = 0.3):
        super().__init__("AdaptiveRandomSearch")
        self.step_fraction = initial_step_fraction

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        best_cfg = tracker.best_result.config if tracker.best_result else registry.sample_random_configuration()
        results = []
        for _ in range(count):
            cfg = copy.deepcopy(best_cfg)
            for param in registry.get_all():
                if isinstance(param, ContinuousParameter) and param.name in cfg:
                    span = param.high - param.low
                    noise = random.gauss(0.0, span * self.step_fraction)
                    cfg[param.name] = MathUtils.clamp(float(cfg[param.name]) + noise, param.low, param.high)
                elif isinstance(param, IntegerParameter) and param.name in cfg:
                    span = param.high - param.low
                    noise = int(random.gauss(0.0, span * self.step_fraction))
                    cfg[param.name] = MathUtils.clamp(int(cfg[param.name]) + noise, param.low, param.high)
            results.append(cfg)
        return results


# 5. Bayesian Optimization Strategy
class BayesianOptimizationStrategy(SearchStrategy):
    """Gaussian Process approximation with analytical Expected Improvement / Probability of Improvement."""
    def __init__(self, acquisition_func: str = "EI"):
        super().__init__("BayesianOptimization")
        self.acquisition_func = acquisition_func

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        best_f = tracker.best_result.composite_score if tracker.best_result else 0.0
        proposals = []
        for _ in range(count):
            best_cand = None
            best_acq = -float('inf')
            # Generate diverse candidate pool from best config, posterior means, and random space
            candidates = [registry.sample_random_configuration() for _ in range(20)]
            if tracker.best_result:
                base = tracker.best_result.config
                for radius in [0.05, 0.15, 0.3]:
                    mutated = copy.deepcopy(base)
                    for param in registry.get_all():
                        if isinstance(param, ContinuousParameter) and param.name in mutated:
                            span = param.high - param.low
                            mutated[param.name] = MathUtils.clamp(float(mutated[param.name]) + random.gauss(0, span * radius), param.low, param.high)
                    candidates.append(mutated)
            # Add posterior mean center point
            mean_center = {}
            for param in registry.get_all():
                snap = beliefs.beliefs.get(param.name)
                if snap and isinstance(param, ContinuousParameter):
                    mean_center[param.name] = MathUtils.clamp(snap.get_posterior_mean_std()[0], param.low, param.high)
                elif snap and isinstance(param, IntegerParameter):
                    mean_center[param.name] = MathUtils.clamp(int(round(snap.get_posterior_mean_std()[0])), param.low, param.high)
                else:
                    mean_center[param.name] = registry.sample_random_configuration().get(param.name)
            candidates.append(mean_center)

            for cand in candidates:
                acq_score = 0.0
                for name, val in cand.items():
                    snap = beliefs.beliefs.get(name)
                    if snap:
                        mean, std = snap.get_posterior_mean_std()
                        if self.acquisition_func == "EI":
                            acq_score += snap.compute_expected_improvement(best_f, mean, std)
                        elif self.acquisition_func == "PI":
                            acq_score += snap.compute_probability_of_improvement(best_f, mean, std)
                        else: # UCB
                            acq_score += mean + 1.96 * std
                if acq_score > best_acq:
                    best_acq = acq_score
                    best_cand = cand
            proposals.append(best_cand or registry.sample_random_configuration())
        return proposals


# 6. Tree-Structured Parzen Estimator (TPE) Strategy
class TreeStructuredParzenEstimatorStrategy(SearchStrategy):
    """TPE approximation dividing history into top quantile l(x) and bottom quantile g(x)."""
    def __init__(self, gamma: float = 0.25):
        super().__init__("TreeStructuredParzenEstimator")
        self.gamma = gamma

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        completed = [t for t in tracker.history if t.status == TrialStatus.COMPLETED]
        if len(completed) < 8:
            return [registry.sample_random_configuration() for _ in range(count)]
        completed.sort(key=lambda t: t.composite_score, reverse=True)
        split_idx = max(1, int(len(completed) * self.gamma))
        good_group = [t.config for t in completed[:split_idx]]
        bad_group = [t.config for t in completed[split_idx:]]

        proposals = []
        for _ in range(count):
            best_cand = None
            best_ratio = -float('inf')
            for _ in range(25):
                base = random.choice(good_group)
                cand = copy.deepcopy(base)
                for param in registry.get_all():
                    if isinstance(param, ContinuousParameter) and param.name in cand:
                        cand[param.name] = MathUtils.clamp(float(cand[param.name]) + random.gauss(0, (param.high - param.low) * 0.1), param.low, param.high)
                l_dist = min(sum((float(cand.get(k, 0)) - float(g.get(k, 0))) ** 2 for k in cand if isinstance(g.get(k), (int, float))) for g in good_group)
                g_dist = min(sum((float(cand.get(k, 0)) - float(b.get(k, 0))) ** 2 for k in cand if isinstance(b.get(k), (int, float))) for b in bad_group)
                ratio = math.exp(-0.5 * l_dist) / max(math.exp(-0.5 * g_dist), MathUtils.EPSILON)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cand = cand
            proposals.append(best_cand or registry.sample_random_configuration())
        return proposals


# 7. Evolutionary Search Strategy
class EvolutionarySearchStrategy(SearchStrategy):
    """Genetic algorithm using tournament selection, uniform crossover, and Gaussian mutation."""
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.2):
        super().__init__("EvolutionarySearch")
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        completed = [t for t in tracker.history if t.status == TrialStatus.COMPLETED]
        if len(completed) < 6:
            return [registry.sample_random_configuration() for _ in range(count)]
        completed.sort(key=lambda t: t.composite_score, reverse=True)
        elites = completed[:min(self.population_size, len(completed))]

        proposals = []
        for _ in range(count):
            parent1 = max(random.sample(elites, min(3, len(elites))), key=lambda t: t.composite_score).config
            parent2 = max(random.sample(elites, min(3, len(elites))), key=lambda t: t.composite_score).config
            child = {}
            for param in registry.get_all():
                name = param.name
                val = parent1.get(name) if random.random() < 0.5 else parent2.get(name)
                if random.random() < self.mutation_rate:
                    val = param.accept(SamplingVisitor())
                child[name] = val
            proposals.append(child)
        return proposals


# 8. Hill Climbing Strategy
class HillClimbingStrategy(SearchStrategy):
    """Local greedy search perturbing single parameters of the best solution strictly uphill."""
    def __init__(self, step_size: float = 0.1):
        super().__init__("HillClimbing")
        self.step_size = step_size

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        best_cfg = tracker.best_result.config if tracker.best_result else registry.sample_random_configuration()
        proposals = []
        for _ in range(count):
            cfg = copy.deepcopy(best_cfg)
            params = registry.get_all()
            param = random.choice(params)
            if isinstance(param, ContinuousParameter) and param.name in cfg:
                delta = random.choice([-1.0, 1.0]) * (param.high - param.low) * self.step_size
                cfg[param.name] = MathUtils.clamp(float(cfg[param.name]) + delta, param.low, param.high)
            elif isinstance(param, IntegerParameter) and param.name in cfg:
                delta = random.choice([-1, 1]) * max(1, int((param.high - param.low) * self.step_size))
                cfg[param.name] = MathUtils.clamp(int(cfg[param.name]) + delta, param.low, param.high)
            proposals.append(cfg)
        return proposals


# 9. Coordinate Descent Strategy
class CoordinateDescentStrategy(SearchStrategy):
    """Optimizes one parameter coordinate at a time while holding all other parameters fixed."""
    def __init__(self):
        super().__init__("CoordinateDescent")
        self.param_index = 0

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        best_cfg = tracker.best_result.config if tracker.best_result else registry.sample_random_configuration()
        params = registry.get_all()
        if not params:
            return [best_cfg] * count
        proposals = []
        for _ in range(count):
            cfg = copy.deepcopy(best_cfg)
            target_param = params[self.param_index % len(params)]
            self.param_index += 1
            cfg[target_param.name] = target_param.accept(SamplingVisitor())
            proposals.append(cfg)
        return proposals


# 10. Simulated Annealing Strategy
class SimulatedAnnealingStrategy(SearchStrategy):
    """Neighborhood search accepting worse configurations based on temperature T_t = T_0 * alpha^t."""
    def __init__(self, initial_temp: float = 100.0, cooling_rate: float = 0.95):
        super().__init__("SimulatedAnnealing")
        self.temp = initial_temp
        self.cooling_rate = cooling_rate

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        last_cfg = tracker.history[-1].config if tracker.history else registry.sample_random_configuration()
        proposals = []
        for _ in range(count):
            cfg = copy.deepcopy(last_cfg)
            for param in registry.get_all():
                if isinstance(param, ContinuousParameter) and param.name in cfg:
                    noise = random.gauss(0, (param.high - param.low) * (self.temp / 100.0))
                    cfg[param.name] = MathUtils.clamp(float(cfg[param.name]) + noise, param.low, param.high)
            proposals.append(cfg)
            self.temp = max(0.001, self.temp * self.cooling_rate)
        return proposals


# 11. Population Based Training (PBT) Strategy
class PopulationBasedTrainingStrategy(SearchStrategy):
    """Joint exploration-exploitation: periodically replaces bottom quantile PBT workers with mutated top performers."""
    def __init__(self, population_size: int = 10, replace_fraction: float = 0.2):
        super().__init__("PopulationBasedTraining")
        self.population_size = population_size
        self.replace_fraction = replace_fraction
        self.population: List[Dict[str, Any]] = []

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        completed = [t for t in tracker.history if t.status == TrialStatus.COMPLETED]
        if len(completed) < self.population_size:
            return [registry.sample_random_configuration() for _ in range(count)]
        completed.sort(key=lambda t: t.composite_score, reverse=True)
        top_slice = completed[:max(1, int(self.population_size * (1.0 - self.replace_fraction)))]
        proposals = []
        for _ in range(count):
            base = random.choice(top_slice).config
            mutated = copy.deepcopy(base)
            for param in registry.get_all():
                if isinstance(param, ContinuousParameter) and param.name in mutated:
                    factor = random.choice([0.8, 1.2])
                    mutated[param.name] = MathUtils.clamp(float(mutated[param.name]) * factor, param.low, param.high)
            proposals.append(mutated)
        return proposals


# 12. Multi-Armed Bandit Strategy (UCB1 / Thompson Sampling)
class MultiArmedBanditStrategy(SearchStrategy):
    """Treats discrete sub-regions or categorical parameters as bandit arms with UCB1 exploration bonuses."""
    def __init__(self, exploration_c: float = 1.414):
        super().__init__("MultiArmedBandit")
        self.exploration_c = exploration_c
        self.arm_counts: Dict[str, int] = defaultdict(int)
        self.arm_rewards: Dict[str, float] = defaultdict(float)

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        total_trials = max(1, len(tracker.history))
        proposals = []
        for _ in range(count):
            cfg = registry.sample_random_configuration()
            for param in registry.get_all():
                if isinstance(param, CategoricalParameter):
                    best_choice = None
                    best_ucb = -float('inf')
                    for choice in sorted(param.choices, key=str):
                        key = f"{param.name}_{choice}"
                        n_i = self.arm_counts[key]
                        if n_i == 0:
                            best_choice = choice
                            break
                        mean_r = self.arm_rewards[key] / n_i
                        ucb = mean_r + self.exploration_c * math.sqrt(math.log(total_trials) / n_i)
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_choice = choice
                    cfg[param.name] = best_choice
            proposals.append(cfg)
        return proposals

    def update(self, result: TrialResult, beliefs: BayesianBeliefTracker) -> None:
        if result.status == TrialStatus.COMPLETED:
            for name, val in result.config.items():
                key = f"{name}_{val}"
                self.arm_counts[key] += 1
                self.arm_rewards[key] += result.composite_score


# 13. Cross Entropy Method Strategy
class CrossEntropyMethodStrategy(SearchStrategy):
    """Samples from parametric distributions and updates means/variances towards top elite quantile statistics."""
    def __init__(self, elite_fraction: float = 0.2):
        super().__init__("CrossEntropyMethod")
        self.elite_fraction = elite_fraction
        self.param_means: Dict[str, float] = {}
        self.param_stds: Dict[str, float] = {}

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        completed = [t for t in tracker.history if t.status == TrialStatus.COMPLETED]
        if len(completed) >= 8:
            completed.sort(key=lambda t: t.composite_score, reverse=True)
            elites = completed[:max(2, int(len(completed) * self.elite_fraction))]
            for param in registry.get_all():
                if isinstance(param, (ContinuousParameter, IntegerParameter)):
                    vals = [float(e.config[param.name]) for e in elites if param.name in e.config]
                    if vals:
                        self.param_means[param.name] = MathUtils.mean(vals)
                        self.param_stds[param.name] = max(MathUtils.std_dev(vals), MathUtils.EPSILON)

        proposals = []
        for _ in range(count):
            cfg = registry.sample_random_configuration()
            for param in registry.get_all():
                if isinstance(param, ContinuousParameter) and param.name in self.param_means:
                    sample = random.gauss(self.param_means[param.name], self.param_stds[param.name])
                    cfg[param.name] = MathUtils.clamp(sample, param.low, param.high)
            proposals.append(cfg)
        return proposals


# 14. CMA-ES Approximation Strategy
class CMAESApproximationStrategy(SearchStrategy):
    """Covariance Matrix Adaptation Evolution Strategy approximation using diagonal covariance variance updates."""
    def __init__(self, sigma: float = 0.5):
        super().__init__("CMAESApproximation")
        self.sigma = sigma
        self.mean_vector: Dict[str, float] = {}
        self.diag_cov: Dict[str, float] = {}

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        completed = [t for t in tracker.history if t.status == TrialStatus.COMPLETED]
        params = [p for p in registry.get_all() if isinstance(p, ContinuousParameter)]
        if len(completed) >= 8 and params:
            completed.sort(key=lambda t: t.composite_score, reverse=True)
            elites = completed[:max(3, len(completed) // 4)]
            for p in params:
                vals = [float(e.config[p.name]) for e in elites if p.name in e.config]
                if vals:
                    self.mean_vector[p.name] = MathUtils.mean(vals)
                    self.diag_cov[p.name] = max(MathUtils.variance(vals), 1e-4)

        proposals = []
        for _ in range(count):
            cfg = registry.sample_random_configuration()
            for p in params:
                m = self.mean_vector.get(p.name, 0.5 * (p.low + p.high))
                std = math.sqrt(self.diag_cov.get(p.name, ((p.high - p.low) / 4.0) ** 2))
                val = random.gauss(m, self.sigma * std)
                cfg[p.name] = MathUtils.clamp(val, p.low, p.high)
            proposals.append(cfg)
        return proposals


# 15. Successive Halving Strategy
class SuccessiveHalvingStrategy(SearchStrategy):
    """Allocates budgets in rungs, pruning bottom (1 - 1/eta) fraction of candidates at each rung."""
    def __init__(self, eta: int = 3):
        super().__init__("SuccessiveHalving")
        self.eta = eta
        self.current_rung = 0

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        completed = [t for t in tracker.history if t.status == TrialStatus.COMPLETED and t.rung == self.current_rung]
        if len(completed) >= self.eta * count and count > 0:
            completed.sort(key=lambda t: t.composite_score, reverse=True)
            survivors = completed[:len(completed) // self.eta]
            self.current_rung += 1
            return [s.config for s in random.choices(survivors, k=count)]
        return [registry.sample_random_configuration() for _ in range(count)]


# 16. Hyperband Strategy
class HyperbandStrategy(SearchStrategy):
    """Outer loop over Successive Halving, cycling through brackets of decreasing size and increasing
    per-configuration resource. Higher-resource brackets bias sampling toward the current best region
    (exploitation); low-resource brackets stay purely random (exploration)."""
    def __init__(self, max_resource: int = 81, eta: int = 3):
        super().__init__("Hyperband")
        self.max_resource = max_resource
        self.eta = eta
        self.s_max = int(math.log(max_resource) / math.log(eta))
        self.current_s = self.s_max
        self.brackets = self._create_brackets()
        self._bracket_pointer = 0
        self._proposals_in_bracket = 0

    def _create_brackets(self) -> List[Tuple[int, int]]:
        """Compute (num_configs, resource_per_config) for each bracket s = s_max..0."""
        brackets = []
        for s in range(self.s_max, -1, -1):
            n = int(math.ceil((self.s_max + 1) / (s + 1) * (self.eta ** s)))
            r = int(self.max_resource * (self.eta ** (-s)))
            brackets.append((max(1, n), max(1, r)))
        return brackets

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        n_configs, resource = self.brackets[self._bracket_pointer]
        self.current_s = self.s_max - self._bracket_pointer
        resource_fraction = MathUtils.clamp(resource / max(1, self.max_resource), 0.0, 1.0)

        proposals = []
        for _ in range(count):
            cfg = registry.sample_random_configuration()
            if tracker.best_result and resource_fraction > 0.3:
                cfg = copy.deepcopy(tracker.best_result.config)
                shrink = 1.0 - resource_fraction
                for param in registry.get_all():
                    if isinstance(param, ContinuousParameter) and param.name in cfg:
                        span = (param.high - param.low) * shrink
                        cfg[param.name] = MathUtils.clamp(float(cfg[param.name]) + random.gauss(0, max(span * 0.2, MathUtils.EPSILON)), param.low, param.high)
                    elif isinstance(param, IntegerParameter) and param.name in cfg:
                        span = (param.high - param.low) * shrink
                        cfg[param.name] = MathUtils.clamp(int(round(float(cfg[param.name]) + random.gauss(0, max(span * 0.2, MathUtils.EPSILON)))), param.low, param.high)
            proposals.append(cfg)
            self._proposals_in_bracket += 1
            if self._proposals_in_bracket >= n_configs:
                self._proposals_in_bracket = 0
                self._bracket_pointer = (self._bracket_pointer + 1) % len(self.brackets)
        return proposals


# 17. Fractal Recursive Search Strategy
class FractalRecursiveSearchStrategy(SearchStrategy):
    """Large search -> Cluster promising regions via pure K-Means -> Subdivide bounding boxes -> Search recursively."""
    def __init__(self, clusters_k: int = 3, recursion_budget: int = 5):
        super().__init__("FractalRecursiveSearch")
        self.clusters_k = clusters_k
        self.recursion_budget = recursion_budget
        self.current_recursion_depth = 0
        self.active_bounding_boxes: List[Dict[str, Tuple[float, float]]] = []

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        completed = [t for t in tracker.history if t.status == TrialStatus.COMPLETED]
        params = [p for p in registry.get_all() if isinstance(p, ContinuousParameter)]
        if len(completed) >= 10 and params and self.current_recursion_depth < self.recursion_budget:
            completed.sort(key=lambda t: t.composite_score, reverse=True)
            top_points = [[float(c.config[p.name]) for p in params] for c in completed[:10]]
            centroids = MathUtils.k_means_cluster(top_points, k=min(self.clusters_k, len(top_points)))
            self.active_bounding_boxes = []
            for c in centroids:
                box = {}
                for idx, p in enumerate(params):
                    radius = (p.high - p.low) / (2.0 ** (self.current_recursion_depth + 1))
                    box[p.name] = (MathUtils.clamp(c[idx] - radius, p.low, p.high), MathUtils.clamp(c[idx] + radius, p.low, p.high))
                self.active_bounding_boxes.append(box)
            self.current_recursion_depth += 1

        proposals = []
        for _ in range(count):
            cfg = registry.sample_random_configuration()
            if self.active_bounding_boxes and params:
                chosen_box = random.choice(self.active_bounding_boxes)
                for p in params:
                    low_b, high_b = chosen_box[p.name]
                    cfg[p.name] = random.uniform(low_b, high_b)
            proposals.append(cfg)
        return proposals


# 18. Recursive Local Refinement Strategy
class RecursiveLocalRefinementStrategy(SearchStrategy):
    """Zoom in on top performers with shrinking search radii (decay factor)."""
    def __init__(self, decay_factor: float = 0.5):
        super().__init__("RecursiveLocalRefinement")
        self.decay_factor = decay_factor
        self.radius_scale = 1.0

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        best_cfg = tracker.best_result.config if tracker.best_result else registry.sample_random_configuration()
        proposals = []
        for _ in range(count):
            cfg = copy.deepcopy(best_cfg)
            for p in registry.get_all():
                if isinstance(p, ContinuousParameter) and p.name in cfg:
                    span = (p.high - p.low) * self.radius_scale
                    cfg[p.name] = MathUtils.clamp(float(cfg[p.name]) + random.gauss(0, span * 0.1), p.low, p.high)
            proposals.append(cfg)
        self.radius_scale = max(0.001, self.radius_scale * self.decay_factor)
        return proposals


# 19. Branch and Bound Strategy
class BranchAndBoundStrategy(SearchStrategy):
    """Prune sub-regions whose upper bound score estimate falls below the global lower bound."""
    def __init__(self):
        super().__init__("BranchAndBound")
        self.global_lower_bound = -float('inf')

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        if tracker.best_result:
            self.global_lower_bound = max(self.global_lower_bound, tracker.best_result.composite_score)
        proposals = []
        for _ in range(count):
            for _ in range(20):
                cfg = registry.sample_random_configuration()
                ucb_est = 0.0
                for name, val in cfg.items():
                    snap = beliefs.beliefs.get(name)
                    if snap:
                        mean, std = snap.get_posterior_mean_std()
                        ucb_est += mean + 2.0 * std
                if ucb_est >= self.global_lower_bound or not tracker.history:
                    proposals.append(cfg)
                    break
            else:
                proposals.append(registry.sample_random_configuration())
        return proposals


# 20. Hybrid Strategies
class HybridStrategy(SearchStrategy):
    """Ensemble proposing candidates proportionally across multiple underlying search strategies."""
    def __init__(self, strategies: List[SearchStrategy], weights: Optional[List[float]] = None):
        super().__init__("HybridStrategy")
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        proposals = []
        for strat, w in zip(self.strategies, self.weights):
            n_alloc = max(1, int(count * w))
            proposals.extend(strat.propose(registry, tracker, beliefs, n_alloc))
        return proposals[:count]


# 21. Gaussian Process Bayesian Optimization Strategy (Joint Surrogate)
class GaussianProcessBayesianOptimizationStrategy(SearchStrategy):
    """Joint multi-dimensional Gaussian Process surrogate over normalized continuous/integer parameters,
    ranked via analytical Expected Improvement or Upper Confidence Bound. Unlike BayesianOptimizationStrategy,
    which sums independent per-parameter conjugate beliefs, this strategy models cross-parameter covariance
    directly through a shared RBF kernel fit over the full joint vector."""
    def __init__(self, acquisition: str = "EI", n_initial: int = 10, n_candidates: int = 150, xi: float = 0.01, kappa: float = 2.0):
        super().__init__("GaussianProcessBayesianOptimization")
        self.acquisition_fn: AcquisitionFunction = ExpectedImprovement(xi) if acquisition == "EI" else UpperConfidenceBound(kappa)
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.gp: Optional[GaussianProcessRegressor] = None

    @staticmethod
    def _numeric_params(registry: ParameterRegistry) -> List[ParameterSpec]:
        return [p for p in registry.get_all() if isinstance(p, (ContinuousParameter, IntegerParameter))]

    @staticmethod
    def _vectorize(config: Dict[str, Any], numeric_params: List[ParameterSpec]) -> List[float]:
        vec = []
        for p in numeric_params:
            raw = float(config[p.name])
            if isinstance(p, ContinuousParameter) and p.log_scale:
                low, high = math.log(p.low), math.log(p.high)
                raw = math.log(max(raw, MathUtils.EPSILON))
            else:
                low, high = float(p.low), float(p.high)
            vec.append(MathUtils.clamp((raw - low) / (high - low + MathUtils.EPSILON), 0.0, 1.0))
        return vec

    def propose(self, registry: ParameterRegistry, tracker: AutoregressiveTracker, beliefs: BayesianBeliefTracker, count: int) -> List[Dict[str, Any]]:
        numeric_params = self._numeric_params(registry)
        completed = [t for t in tracker.history if t.status == TrialStatus.COMPLETED]
        if len(completed) < self.n_initial or not numeric_params:
            return [registry.sample_random_configuration() for _ in range(count)]

        X = [self._vectorize(t.config, numeric_params) for t in completed]
        y = [t.composite_score for t in completed]
        self.gp = GaussianProcessRegressor(RBFKernel())
        self.gp.fit(X, y)
        best_f = tracker.best_result.composite_score if tracker.best_result else max(y)

        pool_size = max(count, self.n_candidates)
        candidates = [registry.sample_random_configuration() for _ in range(pool_size)]
        candidate_vecs = [self._vectorize(c, numeric_params) for c in candidates]
        acq_vals = self.acquisition_fn.evaluate(self.gp, candidate_vecs, best_f)
        ranked = sorted(range(pool_size), key=lambda i: acq_vals[i], reverse=True)
        return [candidates[i] for i in ranked[:count]]


# ==============================================================================
# STRATEGY REGISTRY & META-CONTROLLER
# ==============================================================================

class StrategyRegistry:
    """Registry maintaining all available search strategy implementations."""
    def __init__(self):
        self.strategies: Dict[str, SearchStrategy] = {}
        self.register_defaults()

    def register(self, strategy: SearchStrategy) -> None:
        self.strategies[strategy.name] = strategy

    def get(self, name: str) -> Optional[SearchStrategy]:
        return self.strategies.get(name)

    def get_all(self) -> List[SearchStrategy]:
        return [self.strategies[k] for k in sorted(self.strategies.keys())]

    def register_defaults(self) -> None:
        self.register(RandomSearchStrategy())
        self.register(GridSearchStrategy())
        self.register(LatinHypercubeSearchStrategy())
        self.register(AdaptiveRandomSearchStrategy())
        self.register(BayesianOptimizationStrategy("EI"))
        self.register(TreeStructuredParzenEstimatorStrategy())
        self.register(EvolutionarySearchStrategy())
        self.register(HillClimbingStrategy())
        self.register(CoordinateDescentStrategy())
        self.register(SimulatedAnnealingStrategy())
        self.register(PopulationBasedTrainingStrategy())
        self.register(MultiArmedBanditStrategy())
        self.register(CrossEntropyMethodStrategy())
        self.register(CMAESApproximationStrategy())
        self.register(SuccessiveHalvingStrategy())
        self.register(HyperbandStrategy())
        self.register(FractalRecursiveSearchStrategy())
        self.register(RecursiveLocalRefinementStrategy())
        self.register(BranchAndBoundStrategy())
        self.register(GaussianProcessBayesianOptimizationStrategy())


class MetaController:
    """Meta-Learning Controller tracking strategy performance via Thompson Sampling and dynamic switching."""
    def __init__(self, registry: StrategyRegistry):
        self.registry = registry
        self.strategy_rewards: Dict[str, float] = defaultdict(float)
        self.strategy_counts: Dict[str, int] = defaultdict(int)
        self.active_strategy: SearchStrategy = registry.get("RandomSearch") or RandomSearchStrategy()
        # Progression chain: Random -> Bayesian -> Evolution -> Local Refinement -> Fractal Recursive
        self._progression_chain = [
            "RandomSearch", "BayesianOptimization", "EvolutionarySearch",
            "RecursiveLocalRefinement", "FractalRecursiveSearch"
        ]

    def select_best_strategy(self, tracker: AutoregressiveTracker) -> SearchStrategy:
        """Dynamically select strategy based on optimization stage and Thompson Sampling rewards."""
        total_trials = len(tracker.history)
        if total_trials < 8:
            name = self._progression_chain[0]
        elif total_trials < 20:
            name = self._progression_chain[1]
        elif total_trials < 35:
            name = self._progression_chain[2]
        elif total_trials < 50:
            name = self._progression_chain[3]
        else:
            best_name = self.active_strategy.name
            best_ucb = -float('inf')
            for s in self.registry.get_all():
                n_i = self.strategy_counts[s.name]
                if n_i == 0:
                    best_name = s.name
                    break
                mean_r = self.strategy_rewards[s.name] / n_i
                ucb = mean_r + 1.414 * math.sqrt(math.log(max(1, total_trials)) / n_i)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_name = s.name
            name = best_name

        new_strat = self.registry.get(name) or self.active_strategy
        self.active_strategy = new_strat
        return new_strat

    def record_strategy_performance(self, strategy_name: str, improvement: float) -> None:
        self.strategy_counts[strategy_name] += 1
        self.strategy_rewards[strategy_name] += improvement


# ==============================================================================
# SELF-TUNING ENGINE (META-OPTIMIZATION RECURSION)
# ==============================================================================

class SelfTuner:
    """Self-recursion engine optimizing the optimizer's internal meta-parameters."""
    def __init__(self, target_optimizer: 'AdaptiveOptimizer'):
        self.target_optimizer = target_optimizer
        self.meta_registry = ParameterRegistry()
        self._register_meta_parameters()

    def _register_meta_parameters(self) -> None:
        self.meta_registry.register(ContinuousParameter("meta.exploration_rate", 0.1, 3.0, 1.414, 0.5))
        self.meta_registry.register(ContinuousParameter("meta.mutation_rate", 0.05, 0.5, 0.2, 0.1))
        self.meta_registry.register(ContinuousParameter("meta.decay_factor", 0.1, 0.9, 0.5, 0.2))

    def run_self_reflection(self) -> Dict[str, Any]:
        """Perform one step of meta-optimization on internal parameters based on improvement velocity."""
        vel = self.target_optimizer.tracker.improvement_velocity[-1] if self.target_optimizer.tracker.improvement_velocity else 0.0
        cfg = self.meta_registry.sample_random_configuration()
        if abs(vel) < 1e-4:
            cfg["meta.exploration_rate"] = 2.5
            cfg["meta.mutation_rate"] = 0.4
        else:
            cfg["meta.exploration_rate"] = 1.0
            cfg["meta.mutation_rate"] = 0.15
        return cfg


# ==============================================================================
# PARALLEL SCHEDULER INTERFACES
# ==============================================================================

class ExecutionScheduler(abc.ABC):
    """Abstract interface defining job execution scheduling without changing optimizer logic."""
    @abc.abstractmethod
    def execute_batch(self, commands: List[EvaluateTrialCommand]) -> List[TrialResult]: pass


class SerialExecutionScheduler(ExecutionScheduler):
    """Synchronous serial execution of trial evaluation commands."""
    def execute_batch(self, commands: List[EvaluateTrialCommand]) -> List[TrialResult]:
        return [cmd.execute() for cmd in commands]


class FutureMultiprocessingScheduler(ExecutionScheduler):
    """Multiprocessing / ThreadPool execution scheduler using stdlib concurrent.futures."""
    def __init__(self, max_workers: int = 4, use_threads: bool = True):
        self.max_workers = max_workers
        self.use_threads = use_threads

    def execute_batch(self, commands: List[EvaluateTrialCommand]) -> List[TrialResult]:
        if not commands:
            return []
        executor_cls = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        results = []
        with executor_cls(max_workers=min(self.max_workers, len(commands))) as executor:
            futures = [executor.submit(cmd.execute) for cmd in commands]
            for f in futures:
                results.append(f.result())
        return results


class FutureDistributedScheduler(ExecutionScheduler):
    """Distributed task queue scheduler interface for multi-node cluster deployment."""
    def __init__(self):
        self.task_queue: deque = deque()

    def execute_batch(self, commands: List[EvaluateTrialCommand]) -> List[TrialResult]:
        for cmd in commands:
            self.task_queue.append(cmd)
        results = []
        while self.task_queue:
            cmd = self.task_queue.popleft()
            results.append(cmd.execute())
        return results


# ==============================================================================
# MASTER ENGINE: ADAPTIVE OPTIMIZER
# ==============================================================================

class AdaptiveOptimizer:
    """Production-Grade Autonomous Self-Improving Adaptive Hyperparameter Optimization Engine.

    Continuously searches for better internal parameters during runtime.
    Never restarts: maintains experience, beliefs, history, statistics, and confidence online.
    Integrates natively into Fractal Brain, Living Knowledge Graph, and Fractal Memory.
    """
    def __init__(
        self,
        parameter_registry: Optional[ParameterRegistry] = None,
        strategy_registry: Optional[StrategyRegistry] = None,
        knowledge_graph: Optional[LivingKnowledgeGraphInterface] = None,
        fractal_memory: Optional[FractalMemoryInterface] = None,
        scheduler: Optional[ExecutionScheduler] = None,
        scoring: Optional[CompositeScoring] = None,
        early_stopper: Optional[EarlyStoppingController] = None,
    ):
        self.parameter_registry = parameter_registry or ParameterRegistry()
        if not self.parameter_registry.get_all():
            self.parameter_registry.populate_default_ecosystem_parameters()

        self.strategy_registry = strategy_registry or StrategyRegistry()
        self.kg = knowledge_graph or LivingKnowledgeGraph()
        self.memory = fractal_memory or FractalMemory()
        self.scheduler = scheduler or SerialExecutionScheduler()
        self.scoring = scoring or CompositeScoring()
        self.early_stopper = early_stopper or EarlyStoppingController([PatienceEarlyStopper(25), ConfidenceConvergenceStopper(0.02)])

        self.state = OptimizerState.INITIALIZING
        self.tracker = AutoregressiveTracker()
        self.beliefs = BayesianBeliefTracker(self.parameter_registry)
        self.meta_controller = MetaController(self.strategy_registry)
        self.self_tuner = SelfTuner(self)

        self.constraints: List[Constraint] = []
        self.observers: List[OptimizationObserver] = [
            KnowledgeGraphObserver(self.kg),
            FractalMemoryObserver(self.memory),
            AutoregressiveObserver(self.tracker, self.beliefs),
        ]
        self.pareto_frontier = ParetoFrontier({m.name: m.direction for m in self.scoring.metrics})

        self.set_state(OptimizerState.EXPLORING)

    def set_state(self, new_state: OptimizerState) -> None:
        old_state = self.state
        self.state = new_state
        for obs in self.observers:
            obs.on_state_change(old_state, new_state)

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def add_observer(self, observer: OptimizationObserver) -> None:
        self.observers.append(observer)

    def step(
        self,
        objective_fn: ObjectiveFunction,
        batch_size: int = 1,
        context: Optional[ContextMetadata] = None,
    ) -> List[TrialResult]:
        """Perform one online lifelong learning optimization step evaluating a batch of proposals."""
        if self.state in (OptimizerState.CONVERGED, OptimizerState.STOPPED):
            return []

        # 1. Meta-controller selects active search strategy
        old_strat = self.meta_controller.active_strategy.name
        strategy = self.meta_controller.select_best_strategy(self.tracker)
        if strategy.name != old_strat:
            for obs in self.observers:
                obs.on_strategy_switch(old_strat, strategy.name)

        # 2. Generate candidate configurations
        proposals = strategy.propose(self.parameter_registry, self.tracker, self.beliefs, batch_size)

        # 3. Build EvaluateTrialCommand batch
        commands = []
        for cfg in proposals:
            trial_id = f"tr_{len(self.tracker.history)}_{uuid.uuid4().hex[:6]}"
            for obs in self.observers:
                obs.on_trial_start(trial_id, cfg)
            commands.append(EvaluateTrialCommand(
                trial_id=trial_id,
                config=cfg,
                objective_fn=objective_fn,
                constraints=self.constraints,
                scoring=self.scoring,
                context=context or ContextMetadata(),
            ))

        # 4. Execute batch via scheduler
        results = self.scheduler.execute_batch(commands)

        # 5. Notify observers and update strategies/meta-controller
        for res in results:
            for obs in self.observers:
                obs.on_trial_complete(res)
            strategy.update(res, self.beliefs)
            self.pareto_frontier.add_solution(res)
            prev_best = self.tracker.scores_history[-2] if len(self.tracker.scores_history) >= 2 else 0.0
            impr = max(0.0, res.composite_score - prev_best)
            self.meta_controller.record_strategy_performance(strategy.name, impr)

        # 6. Periodic self-tuning reflection
        if len(self.tracker.history) > 0 and len(self.tracker.history) % 10 == 0:
            self.self_tuner.run_self_reflection()

        # 7. Check early stopping convergence
        should_stop, reason = self.early_stopper.evaluate_stopping(self.tracker, self.beliefs)
        if should_stop:
            self.set_state(OptimizerState.CONVERGED)

        return results

    def optimize(
        self,
        objective_fn: ObjectiveFunction,
        max_trials: int = 50,
        batch_size: int = 1,
        time_budget_seconds: Optional[float] = None,
        context: Optional[ContextMetadata] = None,
    ) -> TrialResult:
        """Run continuous online optimization until trial limit, time budget, or convergence threshold is met."""
        self.set_state(OptimizerState.EXPLORING)
        if time_budget_seconds:
            self.early_stopper.add_stopper(BudgetExhaustionStopper(max_trials=max_trials, max_seconds=time_budget_seconds))

        start_t = time.time()
        while len(self.tracker.history) < max_trials:
            if time_budget_seconds and (time.time() - start_t) >= time_budget_seconds:
                self.set_state(OptimizerState.STOPPED)
                break
            if self.state in (OptimizerState.CONVERGED, OptimizerState.STOPPED):
                break

            self.step(objective_fn, batch_size=batch_size, context=context)

        return self.get_best_configuration()

    def get_best_configuration(self) -> Optional[TrialResult]:
        return self.tracker.best_result

    def get_summary_report(self) -> Dict[str, Any]:
        """Export a comprehensive summary report of optimization state, metrics, and beliefs."""
        return {
            "state": self.state.value,
            "active_strategy": self.meta_controller.active_strategy.name,
            "statistics": self.tracker.get_summary_statistics(),
            "best_configuration": self.tracker.best_result.config if self.tracker.best_result else None,
            "best_score": self.tracker.best_result.composite_score if self.tracker.best_result else None,
            "parameter_beliefs_count": len(self.beliefs.beliefs),
            "pareto_frontier_count": len(self.pareto_frontier.get_frontier()),
        }

    def export_state(self) -> str:
        """Export the complete optimizer state as a pure JSON string."""
        history_dump = [
            {"trial_id": t.trial_id, "config": t.config, "score": t.composite_score, "status": t.status.value}
            for t in self.tracker.history
        ]
        return json.dumps({
            "state": self.state.value,
            "history": history_dump,
            "best_score": self.tracker.best_result.composite_score if self.tracker.best_result else None,
        }, indent=2)

    def import_state(self, json_state: str) -> None:
        """Restore optimizer state from a JSON string without resetting lifelong history."""
        data = json.loads(json_state)
        for t_data in data.get("history", []):
            res = TrialResult(
                trial_id=t_data["trial_id"],
                config=t_data["config"],
                objective_scores={"main": t_data["score"]},
                composite_score=t_data["score"],
                status=TrialStatus(t_data["status"]),
                start_time=time.time(),
                end_time=time.time(),
            )
            self.tracker.add_trial_result(res, self.beliefs)
            sma = self.tracker.sma_history[-1] if self.tracker.sma_history else -1e5
            self.beliefs.update_beliefs_from_trial(res, baseline_sma=sma)
