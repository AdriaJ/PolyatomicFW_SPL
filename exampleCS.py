import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

import pycsou.linop as pl

from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm
from utils import TAPGD, ExplicitMeasurementOperator

from frank_wolfe import VanillaFWSolverForLasso, FullyCorrectiveFWSolverForLasso, \
    PolyatomicFWSolverForLasso

########################

n_sources = 256
alpha = 16
L = alpha * n_sources
grid_size = 512
r = .8  # sampling area (rate of the side length)
psnr = 20
t_max = 10.

seed = None # 8970, 2168

decreasing = True
init_reweighting_prec = 2e-1
multi_spike_threshold = .7

eps = 1e-32
final_reweighting_prec = 1e-4
stopping = 'relative_improvement'  # 'certificate' can be used
lambda_factor = .2
remove = False

########################
start = time.time()
if seed is None:
    seed = np.random.randint(0, 10000)  # 7301, 885
print('Seed: {}'.format(seed))
rng = np.random.default_rng(seed)

margin = round((1 - r) * grid_size / 2)
sources_pos = rng.integers(grid_size - 2 * margin, size=(2, n_sources)) + margin
weights = rng.uniform(3., 6., size=n_sources)
indices = np.ravel_multi_index(sources_pos, (grid_size, grid_size))
sources = np.zeros(grid_size ** 2)  # flatten array, shape (grid_size*grid_size, )
sources[indices] = weights

## Measurements

forward = ExplicitMeasurementOperator(rng.normal(size=(L, grid_size ** 2)), grid_size)
forward.compute_lipschitz_cst(tol=1e-3)
visibilities = forward(sources)
std = np.max(np.abs(visibilities)) * np.exp(-psnr / 10)
noise = rng.normal(0, std, size=L)  # + 1.j * rng.normal(0, std, size=L)
measurements = visibilities + noise

## Lambda

dirty_image = forward.adjoint(measurements)
lambda_ = lambda_factor * np.abs(dirty_image).max()

generation_time = time.time() - start

## APGD

l22_loss = (1 / 2) * SquaredL2Loss(dim=forward.shape[0], data=measurements)
data_fidelity = l22_loss * forward
regularization = lambda_ * L1Norm(dim=forward.shape[1])
apgd = TAPGD(dim=regularization.shape[1], F=data_fidelity, G=regularization, acceleration='CD', verbose=None,
            accuracy_threshold=eps, max_iter=5000, t_max=t_max)
print('APGD')
start = time.time()
res = apgd.iterate()
fista_time = time.time() - start
fista_solution = res[0]['iterand']

## FW Solvers

vfw_solver = VanillaFWSolverForLasso(data=measurements,
                                     forwardOp=forward,
                                     lambda_factor=lambda_factor,
                                     min_iter=0,
                                     max_iter=5000,
                                     stopping_strategy=stopping,
                                     accuracy_threshold=eps,
                                     verbose=None,
                                     remember_iterand=False,
                                     step_size='optimal',
                                     t_max=t_max)

fcfw_solver = FullyCorrectiveFWSolverForLasso(data=measurements,
                                              forwardOp=forward,
                                              lambda_factor=lambda_factor,
                                              min_iter=0,
                                              max_iter=5000,
                                              stopping_strategy=stopping,
                                              accuracy_threshold=eps,
                                              verbose=None,
                                              reweighting_prec=final_reweighting_prec,
                                              remove_positions=remove,
                                              remember_iterand=False,
                                              reweighting='fista',
                                              t_max=t_max)

tpfw_solver = PolyatomicFWSolverForLasso(data=measurements,
                                                forwardOp=forward,
                                                lambda_factor=lambda_factor,
                                                min_iter=0,
                                                max_iter=5000,
                                                stopping_strategy=stopping,
                                                accuracy_threshold=eps,
                                                verbose=None,
                                                init_reweighting_prec=init_reweighting_prec,
                                                final_reweighting_prec=final_reweighting_prec,
                                                multi_spikes_threshold=multi_spike_threshold,
                                                remove_positions=remove,
                                                remember_iterand=False,
                                                decreasing=decreasing,
                                         t_max=t_max)

algos = ['Vanilla FW', 'Fully Corrective FW', 'Polyatomic FW']
solvers = [vfw_solver, fcfw_solver, tpfw_solver]
times = []
res = []
solutions = []

for solver, algo in zip(solvers, algos):
    print(algo)
    start = time.time()
    fw_res = solver.iterate()
    times.append(time.time() - start)
    res.append(fw_res)
    solutions.append(fw_res[0]['iterand'])

## Results

print("Sparsity ratio: {:.3f}%".format(100 * n_sources / (grid_size ** 2)))
print("Measurements over sparsity ratio: {}".format(L / n_sources))
print("Theoretical necessary number of measurements O(k log(N/k)) : {} / {}".format(
    round(n_sources * np.log(grid_size ** 2 / n_sources)), L))
print("FISTA:\n\tRunning time: {}\n\tObjective function: {:.4f}\n\tFinal relative improvement: {}".format(
    fista_time, apgd.objective_functional(fista_solution), apgd.diagnostics['Relative Improvement'].iloc[-1]))
for i in range(len(algos)):
    print(algos[i] +
          ":\n\tRunning time: {}\n\tObjective function: {:.4f}\n\tFinal relative improvement: {}"
          "\n\tDual certificate value: {}".format(
        times[i], solvers[i].diagnostics['Objective Function'].iloc[-1],
        solvers[i].diagnostics['Relative Improvement Objective'].iloc[-1],
        solvers[i].diagnostics['Dual Certificate Value'].iloc[-1]))
print("Error with the sources:")
for i in range(len(algos)):
    print('\t' + algos[i] + ': {}'.format(np.linalg.norm(solutions[i] - sources)/np.linalg.norm(sources)))
print("\tAPGD: {}".format(np.linalg.norm(fista_solution - sources) / np.linalg.norm(sources)))

print('Generation time: {}'.format(generation_time))

sampled_times = np.linspace(0, vfw_solver.t_max, 3000)
labels = ['Vanilla FW', 'FC-FW', 'P-FW']

fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(14, 6))

obj_apgd = np.hstack([.5 * np.linalg.norm(measurements) ** 2, apgd.diagnostics['Objective Function'].values])
interpolated_apgd = np.interp(sampled_times, np.asarray(apgd.times), obj_apgd)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.plot(sampled_times, interpolated_apgd - .99 * interpolated_apgd[-1], label='APGD')
ax2.plot(sampled_times, interpolated_apgd, label='APGD')
for i in range(len(algos)):
    obj_fw = np.hstack([.5 * np.linalg.norm(measurements) ** 2,
                        (solvers[i].diagnostics['Objective Function'].values).astype('float64')])
    interpolated_fw = np.interp(sampled_times, solvers[i].times, obj_fw)
    ax1.plot(sampled_times, interpolated_fw - 0.99*interpolated_apgd[-1], label=labels[i])
    ax2.plot(sampled_times, interpolated_fw, label=labels[i])
ax1.legend()
ax2.legend()
ax1.grid(True, which='both', color='grey', ls='-.', lw=.3)
ax2.grid(True, which='both', color='grey', ls='-.', lw=.3)
ax2.set_title('Objective function value')
ax1.set_title('Difference with the minimum')
fig.suptitle('Grid size: {} - {} sources - {}x measurements'.format(grid_size,
                                                                    n_sources,
                                                                    L//n_sources))
plt.show()

# plt.subplot(132)
# plt.loglog(1 + np.array(scmsfw_solver.times), (obj_fw - 0.99 * obj_fw[-1]).astype('float64'), label='Frank Wolfe')
# plt.loglog(1 + np.array(apgd.times), obj_apgd - .99 * obj_apgd[-1], label='APGD')
# plt.legend()
#
# plt.subplot(133)
# dual_fw = scmsfw_solver.diagnostics['Dual Certificate Value'].values
# plt.scatter(scmsfw_solver.times[:-1], dual_fw)
# plt.hlines([-1., 1.], scmsfw_solver.times[0], scmsfw_solver.times[-2], color='r', ls='--')
# plt.show()


"""norm = mcolors.CenteredNorm()
cmap = cm.bwr

plt.figure(figsize=(14, 6))
plt.subplot(131)
plt.imshow(sources.reshape(grid_size, grid_size), label='Sources', norm=norm, cmap=cmap)
plt.colorbar()

plt.subplot(132)
plt.imshow(fista_solution.reshape(grid_size, grid_size), label='Fista Solution', norm=norm, cmap=cmap)
plt.colorbar()

plt.subplot(133)
plt.imshow(fw_solution.reshape(grid_size, grid_size), label='FW Solution', norm=norm, cmap=cmap)
plt.colorbar()
plt.show()
"""


"""## Support size

plt.figure(figsize=(17, 6))
plt.scatter(range(scmsfw_solver.iter), scmsfw_solver.candidate_new, label='Number of candidates')
plt.scatter(range(scmsfw_solver.iter), scmsfw_solver.actual_new, marker='x', c='r', label='Number of new')
plt.legend()
plt.show()"""

"""
class NonUniformDFT(LinearOperator):
    def __init__(self, L, dim, seed=0):
        # dim is the size of the side of the grid
        self.shape = (dim ** 2, L)
        self.dtype = complex
        self.explicit = False
        self.dim = dim
        self.L = L
        rng = np.random.default_rng(seed)
        self.samples_loc = rng.choice(dim ** 2, L, replace=False)
        self.samples_grid_loc = np.unravel_index(self.samples_loc, (dim, dim))

    def _matvec(self, x):
        xx = x.reshape((self.dim, self.dim))
        xx_f = sfft.fft2(xx, norm="ortho")
        return xx_f.flatten()[self.samples_loc]

    def _rmatvec(self, y):
        yy_f = np.zeros((self.dim, self.dim))
        yy_f[self.samples_grid_loc] = y
        return sfft.ifft2(yy_f, norm="ortho").flatten()
"""

"""
plt.figure(figsize=(14, 6))
plt.subplot(131)
plt.imshow(sources.reshape(grid_size, grid_size), label='Sources', norm=norm, cmap=cmap)
plt.colorbar()

plt.subplot(132)
plt.imshow(dft.adjoint(dft(sources)).reshape(grid_size, grid_size), label='Sources', norm=norm, cmap=cmap)
plt.colorbar()

plt.subplot(133)
mat = np.zeros(dft.shape, dtype=np.complex128)
input = np.eye(dft.shape[1], dtype=np.float64)
for i in range(dft.shape[1]):
    mat[:, i] = dft(input[:, i])
mat = dft.apply_along_axis(input)
plt.imshow((np.abs(mat.transpose().conjugate()@(mat @ sources))).reshape(grid_size, grid_size), label='Sources', norm=norm, cmap=cmap)
plt.colorbar()
plt.show()
"""

"""
plt.figure(figsize=(14, 6))
plt.subplot(131)
plt.imshow(sources.reshape(grid_size, grid_size), label='Sources', norm=norm, cmap=cmap)
plt.colorbar()
plt.subplot(132)
plt.imshow(fista_solution.reshape(grid_size, grid_size), label='Fista Solution', norm=norm, cmap=cmap)
plt.colorbar()
plt.subplot(133)
plt.imshow(sym_fista_solution.reshape(grid_size, grid_size), label='Sym Fista Solution', norm=norm, cmap=cmap)
plt.colorbar()
plt.show()
"""

"""
dual_certificate = forward.adjoint(measurements - forward(fw_solution))
plt.figure()
plt.scatter(np.arange(dual_certificate.shape[0]), dual_certificate)
plt.show()

plt.figure()
plt.scatter(np.arange(dirty_image.shape[0]), dirty_image)
plt.show()
"""


# plt.figure()
# plt.plot(dirty_image)
# plt.show()
