import numpy as np
from typing import Optional, Any
from pandas import DataFrame
from copy import deepcopy
from abc import abstractmethod

from utils import TimedGenericIterativeAlgorithm
import pycsou.core as pcore
import pycsou.linop as pl

from pycsou.func.penalty import L1Norm
from pycsou.func.loss import SquaredL2Loss
from pycsou.opt.proxalgs import APGD


class GenericFWSolverForLasso(TimedGenericIterativeAlgorithm):

    def __init__(self, data: np.ndarray, forwardOp: pcore.linop.LinearOperator, lambda_: Optional[float] = None,
                 lambda_factor: Optional[float] = 0.1, min_iter: int = 10, max_iter: int = 500,
                 stopping_strategy: str = 'certificate', accuracy_threshold: float = 1e-4, verbose: Optional[int] = 10,
                 remove_positions: bool = False, remember_iterand: bool = False, decreasing: bool = False,
                 multi_spikes_threshold: float = .7, multi_spikes: bool = True, reweighting: str = 'ista', t_max: float = None):

        self.data = data
        self.forwardOp = forwardOp
        self.stopping_strategy = stopping_strategy
        self.accuracy_threshold = accuracy_threshold
        self.multi_spikes = multi_spikes
        self.multi_spikes_threshold = multi_spikes_threshold
        self.reweighting = reweighting

        self.remove_positions = remove_positions
        self.decreasing = decreasing

        self.dim = self.forwardOp.shape[1]
        self.x0 = np.zeros(self.dim)
        self.dual_certificate_value = 1 / lambda_factor
        self.new_ind = None
        self.epsilon = None
        self.remember_iterand = remember_iterand
        self.iterand_history = []
        init_iterand = {'iterand': self.x0, 'positions': np.array([], dtype=int)}

        l22_loss = (1 / 2) * SquaredL2Loss(dim=self.forwardOp.shape[0], data=self.data)
        self.data_fidelity = l22_loss * self.forwardOp
        if lambda_ is None:
            lambda_ = lambda_factor * np.abs(self.forwardOp.adjoint(self.data)).max()
        self.lambda_ = lambda_
        self.penalty = self.lambda_ * L1Norm(dim=self.dim)
        objective_functional = self.data_fidelity + self.penalty
        self.bound = np.linalg.norm(self.data) ** 2 / (2 * self.lambda_)

        self.start = None

        if verbose is not None:
            self.candidate_new = []
            self.actual_new = []
        super(GenericFWSolverForLasso, self).__init__(objective_functional=objective_functional,
                                                      init_iterand=init_iterand,
                                                      max_iter=max_iter, min_iter=min_iter,
                                                      accuracy_threshold=accuracy_threshold,
                                                      verbose=verbose, t_max=t_max)

    def update_iterand(self) -> Any:
        self.compute_new_impulse()
        res = self.combine_new_impulse()
        return res

    def compute_new_impulse(self):
        dual_certificate = - self.data_fidelity.gradient(self.old_iterand['iterand']) / self.lambda_
        d = np.abs(dual_certificate)
        if self.multi_spikes:
            maxi = np.max(d)
            if self.iter == 0:
                threshold = self.multi_spikes_threshold * maxi
                self.epsilon = (1 - self.multi_spikes_threshold) * maxi
            else:
                threshold = maxi - (1 / (self.iter + 2)) * self.epsilon
            indices = np.where(d > max(threshold, 1.))[0]
            # print("Threshold: {} / {}".format(threshold, maxi))
            # print('Candidate indices: {}\n'.format(indices.shape))
            self.new_ind = np.setdiff1d(indices, self.old_iterand['positions'], assume_unique=True)
            if self.verbose is not None:
                self.candidate_new.append(indices.shape[0])
                self.actual_new.append(self.new_ind.size)
            if len(self.new_ind) == 0:
                self.new_ind = None
            self.dual_certificate_value = max(dual_certificate.min(),
                                              dual_certificate.max(),
                                              key=abs)
        else:
            self.new_ind = np.argmax(d)
            self.dual_certificate_value = dual_certificate[self.new_ind]
            if self.new_ind in self.old_iterand['positions']:
                self.new_ind = None  # already present position
        if abs(self.dual_certificate_value) < 1.:
            if self.verbose is not None:
                print('Warning, dual certificate lower than 1 at iteration {}'.format(self.iter))

    @abstractmethod
    def combine_new_impulse(self) -> Any:
        pass

    def update_diagnostics(self):
        """
        Dual ceritificate value is computed after iteration

        Returns
        -------

        """
        if self.iter == 0:
            self.diagnostics = DataFrame(
                columns=['Iter', 'Relative Improvement Objective', 'Relative Improvement Iterand',
                         'Dual Certificate Value', 'Objective Function'])
        self.diagnostics.loc[self.iter, 'Iter'] = self.iter
        if np.linalg.norm(self.old_iterand['iterand']) == 0:
            self.diagnostics.loc[self.iter, 'Relative Improvement Iterand'] = np.infty
        else:
            self.diagnostics.loc[self.iter, 'Relative Improvement Iterand'] = np.linalg.norm(
                self.old_iterand['iterand'] - self.iterand['iterand']) / np.linalg.norm(
                self.old_iterand['iterand'])
        self.diagnostics.loc[self.iter, 'Dual Certificate Value'] = self.dual_certificate_value  # before iteration
        self.diagnostics.loc[self.iter, 'Objective Function'] = self.objective_functional(self.iterand['iterand'])
        if self.iter == 0:
            self.diagnostics.loc[self.iter, 'Relative Improvement Objective'] = np.infty
        else:
            self.diagnostics.loc[self.iter, 'Relative Improvement Objective'] = (self.diagnostics.loc[
                                                                                     self.iter - 1,
                                                                                     'Objective Function'] -
                                                                                 self.diagnostics.loc[
                                                                                     self.iter,
                                                                                     'Objective Function']) / \
                                                                                self.diagnostics.loc[
                                                                                    self.iter - 1,
                                                                                    'Objective Function']
        if self.remember_iterand:
            self.iterand_history.append(self.iterand['iterand'])

    def print_diagnostics(self):
        print(dict(self.diagnostics.loc[self.iter]))

    def stopping_metric(self):
        if self.iter == 0:
            return np.infty
        elif self.stopping_strategy == 'relative_improvement':
            return abs(self.diagnostics.loc[self.iter - 1, 'Relative Improvement Objective'])
        elif self.stopping_strategy == 'certificate':
            value = self.diagnostics.loc[self.iter - 1, 'Dual Certificate Value']
            return abs(abs(value) - 1)
        else:
            raise ValueError('Stopping strategy must be in ["relative_improvement", "certificate"]')

    def restricted_support_lasso(self, active_indices: np.ndarray, accuracy: float, x0: np.ndarray = None, d: float = 75.):
        if x0 is None:
            x0 = np.zeros(active_indices.shape)
        injection = pl.sampling.SubSampling(self.dim, active_indices, dtype=float).get_adjointOp()
        restricted_forward = pl.DenseLinearOperator(
            self.forwardOp.mat[:, active_indices])
        restricted_forward.compute_lipschitz_cst(tol=1e-3)
        restricted_data_fidelity = (1 / 2) * SquaredL2Loss(dim=restricted_forward.shape[0], data=self.data) \
                                   * restricted_forward
        # restricted_data_fidelity.lipschitz_cst = self.data_fidelity.lipschitz_cst
        # restricted_data_fidelity.diff_lipschitz_cst = self.data_fidelity.diff_lipschitz_cst
        restricted_regularization = self.lambda_ * L1Norm(dim=restricted_data_fidelity.shape[1])
        if self.reweighting == 'fista':
            acceleration = 'CD'
            tau = None
        elif self.reweighting == 'ista':
            tau = 1.9 / restricted_data_fidelity.diff_lipschitz_cst
            acceleration = None
        else:
            raise ValueError('Reweighting strategy must be in ["fista", "ista"]')
        solver = APGD(dim=restricted_data_fidelity.shape[1], F=restricted_data_fidelity,
                      G=restricted_regularization, x0=x0, tau=tau,
                      acceleration=acceleration, verbose=None, accuracy_threshold=accuracy, d=d, max_iter=2000,
                      min_iter=1)
        return injection(solver.iterate()[0]['iterand'])


class VanillaFWSolverForLasso(GenericFWSolverForLasso):
    def __init__(self, data: np.ndarray, forwardOp: pcore.linop.LinearOperator, lambda_: Optional[float] = None,
                 lambda_factor: Optional[float] = 0.1, min_iter: int = 10, max_iter: int = 500,
                 stopping_strategy: str = 'certificate', accuracy_threshold: float = 1e-4, verbose: Optional[int] = 10,
                 remember_iterand: bool = False, step_size: str = 'optimal', t_max: float = None):

        if step_size in ['optimal', 'regular']:
            self.step_size = step_size
        else:
            raise ValueError("Step size strategy must be in ['optimal', 'regular']")

        super(VanillaFWSolverForLasso, self).__init__(data, forwardOp, lambda_=lambda_, lambda_factor=lambda_factor,
                                                      min_iter=min_iter, max_iter=max_iter,
                                                      stopping_strategy=stopping_strategy,
                                                      accuracy_threshold=accuracy_threshold, verbose=verbose,
                                                      remember_iterand=remember_iterand, multi_spikes=False, t_max=t_max)

    def combine_new_impulse(self) -> Any:
        iterand = deepcopy(self.old_iterand['iterand'])
        if self.new_ind is not None:
            new_positions = np.hstack([self.old_iterand['positions'], self.new_ind])
            if self.step_size == 'optimal':
                gamma = np.dot(self.data_fidelity.gradient(iterand), iterand) + self.lambda_ * (
                    1. * np.linalg.norm(iterand, 1) + (np.abs(self.dual_certificate_value) - 1.) * self.bound)
                gamma /= np.linalg.norm(self.forwardOp.mat[:, self.new_ind] * self.bound * np.sign(
                    self.dual_certificate_value) - self.forwardOp @ iterand, 2) ** 2
            else:
                gamma = 2/(self.iter + 3)
        else:
            new_positions = self.old_iterand['positions']
            if self.step_size == 'optimal':
                gamma = np.dot(self.data_fidelity.gradient(iterand), iterand) + self.lambda_ * np.linalg.norm(iterand, 1)
                gamma /= np.linalg.norm(self.forwardOp @ iterand, 2) ** 2
            else:
                gamma = 2/(self.iter + 3)
        if not 0 < gamma < 1:
            gamma = np.clip(gamma, 0., 1.)
        iterand *= (1 - gamma)
        if self.new_ind is not None:
            iterand[self.new_ind] += gamma * np.sign(self.dual_certificate_value) * self.bound
        return {'iterand': iterand, 'positions': new_positions}


class FullyCorrectiveFWSolverForLasso(VanillaFWSolverForLasso):

    def __init__(self, data: np.ndarray, forwardOp: pcore.linop.LinearOperator, lambda_: Optional[float] = None,
                 lambda_factor: Optional[float] = 0.1, min_iter: int = 10, max_iter: int = 500,
                 stopping_strategy: str = 'certificate', accuracy_threshold: float = 1e-4, verbose: Optional[int] = 10,
                 remember_iterand: bool = False, remove_positions: bool = False, reweighting_prec: float = 1e-4,
                 reweighting: str = 'fista', t_max: float = None):
        self.remove_positions = remove_positions
        self.reweighting_prec = reweighting_prec
        super(FullyCorrectiveFWSolverForLasso, self).__init__(data, forwardOp, lambda_=lambda_,
                                                              lambda_factor=lambda_factor,
                                                              min_iter=min_iter, max_iter=max_iter,
                                                              stopping_strategy=stopping_strategy,
                                                              accuracy_threshold=accuracy_threshold, verbose=verbose,
                                                              remember_iterand=remember_iterand, t_max=t_max)
        self.reweighting = reweighting
        self.last_weight = self.bound


    def combine_new_impulse(self) -> Any:
        iterand = deepcopy(self.old_iterand['iterand'])
        if self.new_ind is not None:
            new_positions = np.unique(np.hstack([self.old_iterand['positions'], self.new_ind]))
            if self.iter > 0 and self.remove_positions:
                active_indices = np.unique(np.hstack([iterand.nonzero()[0], self.new_ind]))
            else:
                active_indices = new_positions
        else:
            new_positions = self.old_iterand['positions']
            if self.iter > 0 and self.remove_positions:
                active_indices = np.unique(iterand.nonzero()[0])
            else:
                active_indices = new_positions

        if active_indices.shape[0] > 1:
            iterand[self.new_ind] = np.sign(self.dual_certificate_value) * self.last_weight
            x0 = iterand[active_indices]
            iterand = self.restricted_support_lasso(active_indices, self.reweighting_prec, x0=x0)
            if self.new_ind is not None:
                self.last_weight = iterand[self.new_ind]
        else:
            tmp = np.zeros(self.dim)
            tmp[active_indices] = 1.
            column = self.forwardOp(tmp)
            iterand[active_indices] = np.dot(self.data, column) / (np.linalg.norm(column, 2) ** 2)
            self.last_weight = iterand[active_indices]
        overvalue = np.abs(iterand) > self.bound
        if overvalue.sum() > 0:
            print("Overvalue at coordinates {}".format(np.arange(overvalue.shape[0])[overvalue]))
            iterand[overvalue] = np.sign(iterand[overvalue]) * self.bound

        return {'iterand': iterand, 'positions': new_positions}


class PolyatomicFWSolverForLasso(GenericFWSolverForLasso):

    def __init__(self, data: np.ndarray, forwardOp: pcore.linop.LinearOperator, lambda_: Optional[float] = None,
                 lambda_factor: Optional[float] = 0.1, min_iter: int = 10, max_iter: int = 500,
                 stopping_strategy: str = 'certificate', accuracy_threshold: float = 1e-4, verbose: Optional[int] = 10,
                 remove_positions: bool = False, remember_iterand: bool = False, final_reweighting_prec: float = 1e-4,
                 init_reweighting_prec: float = .2, decreasing: bool = False, multi_spikes_threshold: float = .7, t_max: float = None):
        self.remove_positions = remove_positions

        self.reweighting_prec = init_reweighting_prec
        self.init_reweighting_prec = init_reweighting_prec
        self.decreasing = decreasing
        self.final_reweighting_prec = final_reweighting_prec

        super(PolyatomicFWSolverForLasso, self).__init__(data, forwardOp, lambda_=lambda_,
                                                                lambda_factor=lambda_factor,
                                                                min_iter=min_iter, max_iter=max_iter,
                                                                stopping_strategy=stopping_strategy,
                                                                accuracy_threshold=accuracy_threshold,
                                                                verbose=verbose,
                                                                remember_iterand=remember_iterand,
                                                                multi_spikes=True,
                                                                multi_spikes_threshold=multi_spikes_threshold,
                                                                reweighting='ista', t_max=t_max)

    def combine_new_impulse(self):
        iterand = deepcopy(self.old_iterand['iterand'])

        if self.new_ind is not None:
            new_positions = np.unique(np.hstack([self.old_iterand['positions'], self.new_ind]))
            if self.iter > 0 and self.remove_positions:
                active_indices = np.unique(np.hstack([iterand.nonzero()[0], self.new_ind]))
            else:
                active_indices = new_positions
        else:
            new_positions = self.old_iterand['positions']
            if self.iter > 0 and self.remove_positions:
                active_indices = np.unique(iterand.nonzero()[0])
            else:
                active_indices = new_positions

        if active_indices.shape[0] > 1:
            x0 = iterand[active_indices]
            iterand = self.restricted_support_lasso(active_indices, self.reweighting_prec, x0=x0)
        else:
            tmp = np.zeros(self.dim)
            tmp[active_indices] = 1.
            column = self.forwardOp(tmp)
            iterand[active_indices] = np.dot(self.data, column) / (np.linalg.norm(column, 2) ** 2)
        overvalue = np.abs(iterand) > self.bound
        if overvalue.sum() > 0:    #Sanity check, never been triggered in practice
            print("Overvalue at coordinates {}".format(np.arange(overvalue.shape[0])[overvalue]))
            iterand[overvalue] = np.sign(iterand[overvalue]) * self.bound

        if self.decreasing:
            self.reweighting_prec = self.init_reweighting_prec / (self.iter + 1)
            self.reweighting_prec = max(self.reweighting_prec, self.final_reweighting_prec)

        return {'iterand': iterand, 'positions': new_positions}
