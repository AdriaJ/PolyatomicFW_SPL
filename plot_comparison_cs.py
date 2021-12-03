import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

import pycsou.linop as pl

from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm
from pycsou.opt.proxalgs import APGD

from frank_wolfe import VanillaFWSolverForLasso, FullyCorrectiveFWSolverForLasso, \
    ThriftyPolytropicFWSolverForLasso

###############################

n_reps = 15
n_sources = [32, 64, 128]
Ls = [16, 64]
grid_size = 128
r = .8  # sampling area (rate of the side length)
psnr = 20

init_seed = None

decreasing = True
init_reweighting_prec = 2e-1
final_reweighting_prec = 1e-4
eps = 1e-32
stopping = 'relative_improvement'  # 'certificate'
lambda_factor = .1
multi_spike_threshold = .7
remove = False

algos = ['FISTA', 'V-FW', 'FC-FW', 'P-FW']
res = {}
sampled_times = np.linspace(0, 4, 1000)
interpolated = {}
###############################

for n in n_sources:
    res[n] = {}
    interpolated[n] = {}
    for factor in Ls:
        print("{} sources - {} measurements".format(n, n * factor))
        L = factor * n
        reps = []
        interpolated_values = [[], [], [], []]

        for _ in range(n_reps):
            if init_seed is None:
                seed = np.random.randint(0, 10000)  # 7301, 885
            else:
                seed = init_seed
            rng = np.random.default_rng(seed)

            # Sources

            margin = round((1 - r) * grid_size / 2)
            sources_pos = rng.integers(0, grid_size - 2 * margin, size=(2, n)) + margin
            weights = rng.uniform(3., 6., size=n)
            indices = np.ravel_multi_index(sources_pos, (grid_size, grid_size))
            sources = np.zeros(grid_size ** 2)  # flatten array, shape (grid_size*grid_size, )
            sources[indices] = weights

            # Measurements

            forward = pl.DenseLinearOperator(rng.normal(size=(L, grid_size ** 2)))
            forward.compute_lipschitz_cst(tol=1e-3)
            visibilities = forward(sources)
            std = np.max(np.abs(visibilities)) * np.exp(-psnr / 10)
            noise = rng.normal(0, std, size=L)
            measurements = visibilities + noise

            # Lambda

            dirty_image = forward.adjoint(measurements)
            lambda_ = lambda_factor * np.abs(dirty_image).max()

            ###############################

            # APGD

            l22_loss = (1 / 2) * SquaredL2Loss(dim=forward.shape[0], data=measurements)
            data_fidelity = l22_loss * forward
            regularization = lambda_ * L1Norm(dim=forward.shape[1])

            apgd = APGD(dim=regularization.shape[1], F=data_fidelity, G=regularization, acceleration='CD', verbose=None,
                        accuracy_threshold=eps, max_iter=5000)

            # FW
            vfw_solver = VanillaFWSolverForLasso(data=measurements,
                                                 forwardOp=forward,
                                                 lambda_factor=lambda_factor,
                                                 min_iter=0,
                                                 max_iter=5000,
                                                 stopping_strategy=stopping,
                                                 accuracy_threshold=eps,
                                                 verbose=None,
                                                 remember_iterand=False)

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
                                                          reweighting='fista')

            tpfw_solver = ThriftyPolytropicFWSolverForLasso(data=measurements,
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
                                                            decreasing=decreasing)

            # Storage of the results

            solvers = [apgd, vfw_solver, fcfw_solver, tpfw_solver]
            times = []
            results = []
            solutions = []
            objectives = []

            for solver, algo in zip(solvers, algos):
                # start = time.time()
                solver_res = solver.iterate()
                # times.append(time.time() - start)
                times.append(solver.times)
                results.append(solver_res)
                solutions.append(solver_res[0]['iterand'])

                objective = np.hstack(
                    [.5 * np.linalg.norm(measurements) ** 2, solver.diagnostics['Objective Function'].values]).astype(
                    'float64')
                objectives.append(objective)
            errors = [np.linalg.norm(solution - sources) / np.linalg.norm(sources) for solution in solutions]
            reps.append({'seed': seed,
                         'objective': objectives,
                         'times': times,
                         'rrmse': min(errors)
                         })
            for i, l in enumerate(interpolated_values):
                l.append(np.interp(sampled_times, np.asarray(times[i]), objectives[i] / objectives[i][0]))
        res[n][factor] = reps
        interpolated[n][factor] = {a: np.asarray(l) for (a, l) in zip(algos, interpolated_values)}

###############################

#import pickle
#pickle.dump((res, interpolated), open("results.p", "wb"))
#res, interpolated = pickle.load(open("results.p", "rb"))


# Plots
plt.style.use('ggplot')
c = ['#E24A33', '#9dcc7a', '#988ED5', '#348ABD']

fig, axes = plt.subplots(3, 2, sharex='all', sharey='all', figsize=(10, 14)) #change
i = 0
for factor in Ls:
    for n in n_sources:
        ax = axes[i % 3][i // 3]    #change
        for j, algo in enumerate(algos):
            values = interpolated[n][factor][algo]
            #ax.set_yscale('log')
            ax.plot(sampled_times, np.median(values, axis=0), label=algo, color=c[j])
            ax.fill_between(sampled_times,
                            np.quantile(values, 0.25, axis=0),
                            np.quantile(values, 0.75, axis=0),
                            alpha=.5, color=c[j])

        errors = [d['rrmse'] for d in res[n][factor]]
        #ax.set_title('mean RRMSE: {:.4f}'.format(sum(errors) / len(errors)))
        ax.grid(True)
        ax.set_ylim([.31, 1.05])
        if i in [2, 5]:    #change
            ax.set_xlabel('Time (s)', size=12)
        if i < 3 :    #change
            ax.set_ylabel('LASSO value', size=14)
        i += 1
axes[0][0].legend(prop={'size': 18}, labelcolor='#737373')
fig.supylabel("Number of sources K: {}".format(n_sources[::-1]))
fig.supxlabel('Number of measurements L: {} * K'.format(Ls))
fig.suptitle('Grid size: {}*{} - {} repetitions'.format(grid_size, grid_size, n_reps), fontsize=20)
plt.show()
