import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft

from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm
from pycsou.linop.sampling import SubSampling
from pycsou.linop import DenseLinearOperator
from utils import TAPGD

from scipy.linalg import svdvals
from scipy.sparse.linalg import svds

from toolsRA import SubSampledDFT, canonical_basis
from frank_wolfe import VanillaFWSolverForLasso, FullyCorrectiveFWSolverForLasso, \
    PolyatomicFWSolverForLasso

########################

n_sources = 64
L = 8 * n_sources
grid_size = 127
r = .8  # sampling area (rate of the side length)
psnr = 60

seed = None  # 8970, 2168 [3614] works!!!

decreasing = True
init_reweighting_prec = 1e-1
multi_spike_threshold = .8

t_max = 10.
eps = 1e-32
final_reweighting_prec = 1e-4
stopping = 'relative_improvement'  # 'certificate'
lambda_factor = .1
remove = False

########################

if seed is None:
    seed = np.random.randint(0, 10000)  # 7301, 885
print('Seed: {}'.format(seed))
rng = np.random.default_rng(seed)

# Source
margin = round((1 - r) * grid_size / 2)
# sources_pos = np.stack([margin + rng.choice(grid_size - 2 * margin, size=n_sources, replace=True),
#                         margin + rng.choice(grid_size - 2 * margin, size=n_sources, replace=True)])
sources_pos = rng.integers(grid_size - 2 * margin, size=(2, n_sources)) + margin
weights = rng.uniform(3., 6., size=n_sources)
indices = np.ravel_multi_index(sources_pos, (grid_size, grid_size))
sources = np.zeros(grid_size ** 2)  # flatten array, shape (grid_size*grid_size, )
sources[indices] = weights

# Sampling
# 1D array of size L
sampled_frequencies = rng.choice(np.arange(grid_size * (grid_size // 2 + 1)), size=L, replace=False)
forward = SubSampledDFT(grid_size, sampled_frequencies)
noiseless_visi = forward(sources)
std = np.max(np.abs(noiseless_visi)) * np.exp(-psnr / 10)
noise = rng.normal(0, std, size=L) + 1.j * rng.normal(0, std, size=L)
measurements = noiseless_visi + noise

########################
# Solving

# APGD
dirty_image = forward.adjoint(measurements)
lambda_ = lambda_factor * np.abs(dirty_image).max()

l22_loss = (1 / 2) * SquaredL2Loss(dim=forward.shape[0], data=measurements)
data_fidelity = l22_loss * forward
regularization = lambda_ * L1Norm(dim=forward.shape[1])
apgd = TAPGD(dim=regularization.shape[1], F=data_fidelity, G=regularization, acceleration=None, verbose=None,
             accuracy_threshold=eps, max_iter=5000, t_max=t_max)
print("APGD")
res = apgd.iterate()
fista_time = apgd.elapsed_time
fista_solution = res[0]['iterand']

## FW Solvers

vfw_solver = VanillaFWSolverForLasso(data=measurements,
                                     forwardOp=forward,
                                     lambda_factor=lambda_factor,
                                     min_iter=10,
                                     max_iter=5000,
                                     stopping_strategy=stopping,
                                     accuracy_threshold=eps,
                                     verbose=50,
                                     remember_iterand=False,
                                     step_size='optimal',
                                     t_max=t_max)

#forward = SubSampledDFT(grid_size, sampled_frequencies)
fcfw_solver = FullyCorrectiveFWSolverForLasso(data=measurements,
                                              forwardOp=forward,
                                              lambda_factor=lambda_factor,
                                              min_iter=10,
                                              max_iter=5000,
                                              stopping_strategy=stopping,
                                              accuracy_threshold=eps,
                                              verbose=50,
                                              reweighting_prec=final_reweighting_prec,
                                              remove_positions=remove,
                                              remember_iterand=False,
                                              reweighting='fista',
                                              t_max=t_max)

#forward = SubSampledDFT(grid_size, sampled_frequencies)
tpfw_solver = PolyatomicFWSolverForLasso(data=measurements,
                                         forwardOp=forward,
                                         lambda_factor=lambda_factor,
                                         min_iter=10,
                                         max_iter=5000,
                                         stopping_strategy=stopping,
                                         accuracy_threshold=eps,
                                         verbose=50,
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
    fw_res = solver.iterate()
    times.append(solver.elapsed_time)
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
    print('\t' + algos[i] + ': {}'.format(np.linalg.norm(solutions[i] - sources) / np.linalg.norm(sources)))
print("\tAPGD: {}".format(np.linalg.norm(fista_solution - sources) / np.linalg.norm(sources)))


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
    ax1.plot(sampled_times, interpolated_fw - 0.99 * interpolated_apgd[-1], label=labels[i])
    ax2.plot(sampled_times, interpolated_fw, label=labels[i])
ax1.legend()
ax2.legend()
ax1.grid(True, which='both', color='grey', ls='-.', lw=.3)
ax2.grid(True, which='both', color='grey', ls='-.', lw=.3)
ax2.set_title('Objective function value')
ax1.set_title('Difference with the minimum')
fig.suptitle('Grid size: {} - {} sources - {}x measurements'.format(grid_size,
                                                                    n_sources,
                                                                    L // n_sources))
plt.show()

# Comparison of the reconstructions
'''fig = plt.figure(figsize=(12, 8))
vmin = min(sources.min(), fista_solution.min(), min(sol.min() for sol in solutions))
vmax = max(sources.max(), fista_solution.max(), max(sol.max() for sol in solutions))
ax = fig.add_subplot(231)
im = ax.imshow(sources.reshape((grid_size, grid_size)), vmin=vmin, vmax=vmax, origin='lower')
fig.colorbar(im, ax=ax)
ax.set_title('Sources')
ax = fig.add_subplot(232)
im = ax.imshow(fista_solution.reshape((grid_size, grid_size)), vmin=vmin, vmax=vmax, origin='lower')
fig.colorbar(im, ax=ax)
ax.set_title('Fista solution')
for i in range(3):
    ax = fig.add_subplot(2, 3, 4+i)
    im = ax.imshow(solutions[i].reshape((grid_size, grid_size)), vmin=vmin, vmax=vmax, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title(f'{algos[i]} solution')

plt.suptitle("FISTA reconstruction")
plt.show()'''