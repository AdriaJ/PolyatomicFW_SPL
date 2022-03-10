import time

import numpy as np
import matplotlib.pyplot as plt

from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm

from utils import TAPGD, ExplicitMeasurementOperator

from toolsRA import SubSampledDFT
from frank_wolfe import PolyatomicFWSolverForLasso

import pickle, os.path

########################

n_sources = 200
L = 8 * n_sources
grid_size = 255
r = .8  # sampling area (rate of the side length)
psnr = 50

seed = None  # 8970, 2168 [3614] works!!!

decreasing = True
# init_reweighting_prec = 1e-1
# multi_spike_threshold = .8

precisions_init = [1e-2, 5e-3, 1e-3]
thresholds = [.6, .65, .7]

t_max = 4.
eps = 1e-32
final_reweighting_prec = 5e-5
stopping = 'relative_improvement'  # 'certificate'
lambda_factor = .15
remove = True
remember = False

########################

if seed is None:
    seed = np.random.randint(0, 10000)  # 7301, 885
print('Seed: {}'.format(seed))
rng = np.random.default_rng(seed)

# Source
margin = round((1 - r) * grid_size / 2)
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

# Compute the explicit DFT matrix

if os.path.isfile('dft'+str(grid_size)+'.p'):
    dft_mat = pickle.load(open('dft'+str(grid_size)+'.p', 'rb'))
else:
    start = time.time()
    dft_mat = forward(np.eye(forward.shape[1]))
    matrix_time = time.time() - start
    print(f'Computation time of the DFT matrix: {matrix_time}')
    pickle.dump(dft_mat, open('dft'+str(grid_size)+'.p', 'wb'))
forward = ExplicitMeasurementOperator(dft_mat, side_length=grid_size)
forward.diff_lipschitz_cst = 1.
forward.lipschitz_cst = 1.

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

## FW Solver

solvers = []

for prec in precisions_init:
    for thresh in thresholds:
        forward = SubSampledDFT(grid_size, sampled_frequencies)
        solver = PolyatomicFWSolverForLasso(data=measurements,
                                            forwardOp=forward,
                                            lambda_=lambda_,
                                            min_iter=4,
                                            max_iter=5000,
                                            stopping_strategy=stopping,
                                            accuracy_threshold=eps,
                                            verbose=None,
                                            init_reweighting_prec=prec,
                                            final_reweighting_prec=final_reweighting_prec,
                                            multi_spikes_threshold=thresh,
                                            remove_positions=remove,
                                            remember_iterand=False,
                                            decreasing=decreasing,
                                            t_max=t_max,
                                            remember_candidates_count=remember,
                                            remember_iterations_reweighting=remember)
        solvers.append(solver)

times = []
res = []
solutions = []

for i, solver in enumerate(solvers):
    print("Solver number {}".format(i))
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
for i in range(len(solvers)):
    print("Context {} -> init reweighting prec {} - init threshold {}".format(i, precisions_init[i // len(thresholds)],
                                                                              thresholds[i % len(thresholds)]) +
          ":\n\tRunning time: {:.4f}\n\tObjective function: {:.4f}\n\tFinal relative improvement: {:.4f}"
          "\n\tDual certificate value: {:.4f}".format(
              times[i], solvers[i].diagnostics['Objective Function'].iloc[-1],
              solvers[i].diagnostics['Relative Improvement Objective'].iloc[-1],
              solvers[i].diagnostics['Dual Certificate Value'].iloc[-1]))
# print("Error with the sources:")
# for i in range(len(algos)):
#     print('\t' + algos[i] + ': {}'.format(np.linalg.norm(solutions[i] - sources) / np.linalg.norm(sources)))
# print("\tAPGD: {}".format(np.linalg.norm(fista_solution - sources) / np.linalg.norm(sources)))

solution_sparsity = [(np.abs(sol) > 1e-6).sum() for sol in solutions]
print('Final solution sparsity: {}  //  APGD sparsity: {}'.format(solution_sparsity, (np.abs(fista_solution) > 1e-6).sum()))
print('\tDimension: {} - Input sparsity: {}'.format(grid_size*grid_size, n_sources))
print("Total elapsed times: ", ['{:.4f}'.format(sol.elapsed_time) for sol in solvers])
print("Number of iterations: {}".format([sol.iter for sol in solvers]))
print("Final reweighting precision: ", ['{:.1e}'.format(sol.reweighting_prec) for sol in solvers])

sampled_times = np.linspace(0, t_max, 3000)

fig = plt.figure(figsize=(len(thresholds) * 4, len(precisions_init) * 4))
obj_apgd = np.hstack([.5 * np.linalg.norm(measurements) ** 2, apgd.diagnostics['Objective Function'].values])
interpolated_apgd = np.interp(sampled_times, np.asarray(apgd.times), obj_apgd)
for i in range(len(solvers)):
    ax = fig.add_subplot(len(precisions_init), len(thresholds), i + 1)
    ax.set_yscale('log')
    ax.plot(sampled_times, interpolated_apgd, label='APGD')
    obj_fw = np.hstack([.5 * np.linalg.norm(measurements) ** 2,
                        (solvers[i].diagnostics['Objective Function'].values).astype('float64')])
    interpolated_fw = np.interp(sampled_times, solvers[i].times, obj_fw)
    ax.plot(sampled_times, interpolated_fw, label="PFW")
    ax.legend()
    ax.grid(True, which='both', color='grey', ls='-.', lw=.3)
    ax.set_title('Init precision: {} - Threshold: {}'.format(precisions_init[i // len(thresholds)],
                                                             thresholds[i % len(thresholds)]))
    if remember:
        ax_bis = ax.twinx()
        ax_bis.scatter(solvers[i].times[1:], solvers[i].candidates_count, c='r', marker='+', s=20)
        ax_bis.scatter(solvers[i].times[1:], solvers[i].iterations_reweighting, c='b', marker='x', s=20)

fig.suptitle('Grid size: {} - {} sources - {}x measurements - psnr {}dB'.format(grid_size,
                                                                                n_sources,
                                                                                L // n_sources,
                                                                                psnr))
plt.show()

# plt.figure()
# plt.subplot(121)
# plt.plot(apgd.diagnostics['Relative Improvement'])
# plt.subplot(122)
# plt.plot(apgd.diagnostics['Relative Improvement'])
# plt.yscale('log')
# plt.show()