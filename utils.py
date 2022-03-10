r"""
This module provides the base class for iterative algorithms.
"""
import pycsou.core.linop
from pycsou.core.map import Map
from typing import Optional, Tuple, Any
from abc import abstractmethod
from copy import deepcopy
import time as t
import numpy as np
from pandas import DataFrame

from pycsou.core.solver import GenericIterativeAlgorithm
from pycsou.core.functional import ProximableFunctional
from pycsou.core.map import DifferentiableMap
from pycsou.linop import DenseLinearOperator

class TimedGenericIterativeAlgorithm(GenericIterativeAlgorithm):
    r"""
    Updated base class for iterative algorithms, integrating a maximal runtime.

    Any instance/subclass of this class must at least implement the abstract methods ``update_iterand``, ``print_diagnostics``
    ``update_diagnostics`` and ``stopping_metric``.
    """

    def __init__(self, objective_functional: Map, init_iterand: Any, max_iter: int = 500, min_iter: int = 10,
                 accuracy_threshold: float = 1e-3, verbose: Optional[int] = None, t_max: float = None):
        r"""
        Parameters
        ----------
        objective_functional: Map
            Objective functional to minimise.
        init_iterand: Any
            Initial guess for warm start.
        max_iter: int
            Maximum number of iterations.
        min_iter: int
            Minimum number of iterations.
        accuracy_threshold: float
            Accuracy threshold for stopping criterion.
        verbose: int
            Print diagnostics every ``verbose`` iterations. If ``None`` does not print anything.
        """
        self.start_time = None
        if t_max is None:
            t_max = 5.
        self.t_max = t_max
        self.elapsed_time = 0.
        self.times = [0.]
        super(TimedGenericIterativeAlgorithm, self).__init__(objective_functional, init_iterand, max_iter, min_iter, accuracy_threshold, verbose)

    def iterate(self) -> Any:
        r"""
        Run the algorithm.

        Returns
        -------
        Any
            Algorithm outcome.
        """
        self.start_time = t.time()
        self.old_iterand = deepcopy(self.init_iterand)
        while ((self.iter <= self.max_iter) and (self.stopping_metric() > self.accuracy_threshold) and self.elapsed_time < self.t_max) or (
                self.iter <= self.min_iter):
            self.iterand = self.update_iterand()
            self.elapsed_time = t.time() - self.start_time
            self.update_diagnostics()
            if self.verbose is not None:
                if self.iter % self.verbose == 0:
                    self.print_diagnostics()
            self.old_iterand = deepcopy(self.iterand)
            self.iter += 1
            self.times.append(self.elapsed_time)
        self.converged = self.stopping_metric() < self.accuracy_threshold
        self.iterand = self.postprocess_iterand()
        return self.iterand, self.converged, self.diagnostics

    def postprocess_iterand(self) -> Any:
        return self.iterand

    def reset(self):
        r"""
        Reset the algorithm.
        """
        self.iter = 0
        self.iterand = None

    def iterates(self, n: int) -> Tuple:
        r"""
        Generator allowing to loop through the n first iterates.

        Useful for debugging/plotting purposes.

        Parameters
        ----------
        n: int
            Max number of iterates to loop through.
        """
        self.reset()
        for i in range(n):
            self.iterand = self.update_iterand()
            self.iter += 1
            yield self.iterand

    @abstractmethod
    def update_iterand(self) -> Any:
        r"""
        Update the iterand.

        Returns
        -------
        Any
            Result of the update.
        """
        pass

    @abstractmethod
    def print_diagnostics(self):
        r"""
        Print diagnostics.
        """
        pass

    @abstractmethod
    def stopping_metric(self):
        r"""
        Stopping metric.
        """
        pass

    @abstractmethod
    def update_diagnostics(self):
        """Update the diagnostics."""
        pass


class TimedAcceleratedProximalGradientDescent(TimedGenericIterativeAlgorithm):
    r"""
    Accelerated proximal gradient descent.

    This class is also accessible via the alias ``APGD()``.

    Notes
    -----
    The *Accelerated Proximal Gradient Descent (APGD)* method can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower semicontinuous* and *convex function* with a *simple proximal operator*.
    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:** the algorithm is still valid if one or more of the terms :math:`\mathcal{F}` or :math:`\mathcal{G}` is zero.

    **Remark 2:**  The convergence is guaranteed for step sizes :math:`\tau\leq 1/\beta`. Without acceleration, APGD can be seen
    as a PDS method with :math:`\rho=1`. The various acceleration schemes are described in [APGD]_.
    For :math:`0<\tau\leq 1/\beta` and Chambolle and Dossal's acceleration scheme (``acceleration='CD'``), APGD achieves the following (optimal) *convergence rates*:

    .. math::

       \lim\limits_{n\rightarrow \infty} n^2\left\vert \mathcal{J}(\mathbf{x}^\star)- \mathcal{J}(\mathbf{x}_n)\right\vert=0\qquad \&\qquad \lim\limits_{n\rightarrow \infty} n^2\Vert \mathbf{x}_n-\mathbf{x}_{n-1}\Vert^2_\mathcal{X}=0,


    for *some minimiser* :math:`{\mathbf{x}^\star}\in\arg\min_{\mathbf{x}\in\mathbb{R}^N} \;\left\{\mathcal{J}(\mathbf{x}):=\mathcal{F}(\mathbf{x})+\mathcal{G}(\mathbf{x})\right\}`.
    In other words, both the objective functional and the APGD iterates :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}` converge at a rate :math:`o(1/n^2)`. In comparison
    Beck and Teboule's acceleration scheme (``acceleration='BT'``) only achieves a convergence rate of :math:`O(1/n^2)`.
    Significant practical *speedup* can moreover be achieved for values of :math:`d` in the range  :math:`[50,100]`  [APGD]_.

    Examples
    --------
    Consider the *LASSO problem*:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda \|\mathbf{x}\|_1,

    with :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda>0.` This problem can be solved via APGD with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2` and :math:`\mathcal{G}(\mathbf{x})=\lambda \|\mathbf{x}\|_1`. We have:

    .. math::

       \mathbf{\nabla}\mathcal{F}(\mathbf{x})=\mathbf{G}^T(\mathbf{G}\mathbf{x}-\mathbf{y}), \qquad  \text{prox}_{\lambda\|\cdot\|_1}(\mathbf{x})=\text{soft}_\lambda(\mathbf{x}).

    This yields the so-called *Fast Iterative Soft Thresholding Algorithm (FISTA)*, whose convergence is guaranteed for :math:`d>2` and :math:`0<\tau\leq \beta^{-1}=\|\mathbf{G}\|_2^{-2}`.

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.func.loss import SquaredL2Loss
       from pycsou.func.penalty import L1Norm
       from pycsou.linop.base import DenseLinearOperator
       from pycsou.opt.proxalgs import APGD

       rng = np.random.default_rng(0)
       G = DenseLinearOperator(rng.standard_normal(15).reshape(3,5))
       G.compute_lipschitz_cst()
       x = np.zeros(G.shape[1])
       x[1] = 1
       x[-2] = -1
       y = G(x)
       l22_loss = (1/2) * SquaredL2Loss(dim=G.shape[0], data=y)
       F = l22_loss * G
       lambda_ = 0.9 * np.max(np.abs(F.gradient(0 * x)))
       G = lambda_ * L1Norm(dim=G.shape[1])
       apgd = APGD(dim=G.shape[1], F=F, G=G, acceleration='CD', verbose=None)
       estimate, converged, diagnostics = apgd.iterate()
       plt.figure()
       plt.stem(x, linefmt='C0-', markerfmt='C0o')
       plt.stem(estimate['iterand'], linefmt='C1--', markerfmt='C1s')
       plt.legend(['Ground truth', 'LASSO Estimate'])
       plt.show()

    See Also
    --------
    :py:class:`~pycsou.opt.proxalgs.APGD`
    """

    def __init__(self, dim: int, F: Optional[DifferentiableMap] = None, G: Optional[ProximableFunctional] = None,
                 tau: Optional[float] = None, acceleration: Optional[str] = 'CD', beta: Optional[float] = None,
                 x0: Optional[np.ndarray] = None, max_iter: int = 500, min_iter: int = 10,
                 accuracy_threshold: float = 1e-3, verbose: Optional[int] = 1, d: float = 75., t_max: float = None):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the objective functional's domain.
        F: Optional[DifferentiableMap]
            Differentiable map :math:`\mathcal{F}`.
        G: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{G}`.
        tau: Optional[float]
            Primal step size.
        acceleration: Optional[str] [None, 'BT', 'CD']
            Which acceleration scheme should be used (`None` for no acceleration).
        beta: Optional[float]
            Lipschitz constant :math:`\beta` of the derivative of :math:`\mathcal{F}`.
        x0: Optional[np.ndarray]
            Initial guess for the primal variable.
        max_iter: int
            Maximal number of iterations.
        min_iter: int
            Minimal number of iterations.
        accuracy_threshold: float
            Accuracy threshold for stopping criterion.
        verbose: int
            Print diagnostics every ``verbose`` iterations. If ``None`` does not print anything.
        d: float
            Parameter :math:`d` for Chambolle and Dossal's acceleration scheme (``acceleration='CD'``).
        """
        self.dim = dim
        self.acceleration = acceleration
        self.d = d
        if isinstance(F, DifferentiableMap):
            if F.shape[1] != dim:
                raise ValueError(f'F does not have the proper dimension: {F.shape[1]}!={dim}.')
            else:
                self.F = F
            if F.diff_lipschitz_cst < np.infty:
                self.beta = self.F.diff_lipschitz_cst if beta is None else beta
            elif (beta is not None) and isinstance(beta, Number):
                self.beta = beta
            else:
                raise ValueError('F must be a differentiable functional with Lipschitz-continuous gradient.')
        elif F is None:
            self.F = NullDifferentiableFunctional(dim=dim)
            self.beta = 0
        else:
            raise TypeError(f'F must be of type {DifferentiableMap}.')

        if isinstance(G, ProximableFunctional):
            if G.dim != dim:
                raise ValueError(f'G does not have the proper dimension: {G.dim}!={dim}.')
            else:
                self.G = G
        elif G is None:
            self.G = NullProximableFunctional(dim=dim)
        else:
            raise TypeError(f'G must be of type {ProximableFunctional}.')

        if tau is not None:
            self.tau = tau
        else:
            self.tau = self.set_step_size()

        if x0 is not None:
            self.x0 = np.asarray(x0)
        else:
            self.x0 = self.initialize_iterate()
        objective_functional = self.F + self.G
        init_iterand = {'iterand': self.x0, 'past_aux': 0 * self.x0, 'past_t': 1}
        super(TimedAcceleratedProximalGradientDescent, self).__init__(objective_functional=objective_functional,
                                                                 init_iterand=init_iterand,
                                                                 max_iter=max_iter, min_iter=min_iter,
                                                                 accuracy_threshold=accuracy_threshold,
                                                                 verbose=verbose, t_max=t_max)

    def set_step_size(self) -> float:
        r"""
        Set the step size to its largest admissible value :math:`1/\beta`.

        Returns
        -------
        Tuple[float, float]
            Largest admissible step size.
        """
        return 1 / self.beta

    def initialize_iterate(self) -> np.ndarray:
        """
        Initialize the iterand to zero.

        Returns
        -------
        np.ndarray
            Zero-initialized iterand.
        """
        return np.zeros(shape=(self.dim,), dtype=np.float)

    def update_iterand(self) -> Any:
        if self.iter == 0:
            x, x_old, t_old = self.init_iterand.values()
        else:
            x, x_old, t_old = self.iterand.values()
        x_temp = self.G.prox(x - self.tau * self.F.gradient(x), tau=self.tau)
        if self.acceleration == 'BT':
            t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        elif self.acceleration == 'CD':
            t = (self.iter + self.d) / self.d
        else:
            t = t_old = 1
        a = (t_old - 1) / t
        x = x_temp + a * (x_temp - x_old)
        iterand = {'iterand': x, 'past_aux': x_temp, 'past_t': t}
        return iterand

    def print_diagnostics(self):
        print(dict(self.diagnostics.loc[self.iter]))

    def stopping_metric(self):
        if self.iter == 0:
            return np.infty
        else:
            return self.diagnostics.loc[self.iter - 1, 'Relative Improvement']

    def update_diagnostics(self):
        if self.iter == 0:
            self.diagnostics = DataFrame(
                columns=['Iter', 'Relative Improvement'])
        self.diagnostics.loc[self.iter, 'Iter'] = self.iter
        self.diagnostics.loc[self.iter, 'Objective Function'] = self.objective_functional(self.iterand['iterand'])
        if np.linalg.norm(self.old_iterand['iterand']) == 0:
            self.diagnostics.loc[self.iter, 'Relative Improvement'] = np.infty
        else:
            self.diagnostics.loc[self.iter, 'Relative Improvement'] = np.linalg.norm(
                self.old_iterand['iterand'] - self.iterand['iterand']) / np.linalg.norm(
                self.old_iterand['iterand'])

TAPGD = TimedAcceleratedProximalGradientDescent

class ExplicitMeasurementOperator(DenseLinearOperator):
    def __init__(self, array: np.ndarray, side_length: int):
        super(ExplicitMeasurementOperator, self).__init__(array)
        self.side_length = side_length

    def adjoint(self, y):
        return super(ExplicitMeasurementOperator, self).adjoint(y).real

    def get_restricted_operator(self, column_indices) -> DenseLinearOperator:
        op = DenseLinearOperator(self.mat[:, column_indices])
        op.lipschitz_cst = self.lipschitz_cst
        op.diff_lipschitz_cst = self.diff_lipschitz_cst
        return op
