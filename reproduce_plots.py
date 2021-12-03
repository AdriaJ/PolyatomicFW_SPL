import numpy as np
import matplotlib.pyplot as plt
import pickle

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
sampled_times = np.linspace(0, 4, 1000)

################################

res, interpolated = pickle.load(open("results.p", "rb"))

plt.style.use('ggplot')
c = ['#E24A33', '#9dcc7a', '#988ED5', '#348ABD']
i = 0
for factor in Ls:
    for n in n_sources:
        i += 1
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_ylim([.31, 1.05])
        for j, algo in enumerate(algos):
            values = interpolated[n][factor][algo]
            #ax.set_yscale('log')
            ax.plot(sampled_times, np.median(values, axis=0), label=algo, color=c[j])
            ax.fill_between(sampled_times,
                            np.quantile(values, 0.25, axis=0),
                            np.quantile(values, 0.75, axis=0),
                            alpha=.5,
                            color=c[j])
        errors = [d['rrmse'] for d in res[n][factor]]
        if i == 1:
            ax.legend(prop={'size': 18}, labelcolor='#737373')
        if i in [2, 4, 6]:
            ax.tick_params(axis='y', labelcolor='white')
        if i in [1, 2, 3, 4]:
            ax.tick_params(axis='x', labelcolor='white')

        if i > 4:
            ax.set_xlabel('Time (s)', size=12)
        if i % 2 == 1:
            ax.set_ylabel('LASSO value', size=14)
        plt.savefig('figures/fig'+str(i)+'.png', format='png', dpi=1000, bbox_inches='tight')
        plt.close(fig)
