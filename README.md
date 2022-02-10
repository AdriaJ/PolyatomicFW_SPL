# Polyatomic Frank-Wolfe for the LASSO

This repository provides the code to run and reproduce the results for the IEEE Signal Processing Letters paper "A Fast and Scalable Polyatomic Frank-Wolfe Algorithm for the LASSO", from Adrian Jarret, Julien Fageot and Matthieu Simeoni.

The implementation is based on the optimization package [Pycsou](https://github.com/matthieumeo/pycsou)

## Installation

The code has been developed and tested on x86_64 systems running Linux, with Python 3.9.


### Dependencies

The notebook extra dependencies are listed in the file ``requirements.txt``.
It is recommended to install those extra dependencies in an Anaconda environment [Miniconda](https://conda.io/miniconda.html) or
[Anaconda](https://www.anaconda.com/download/#linux). 

```bash
>> git clone https://github.com/AdriaJ/PolyatomicFW_SPL
>> cd PolyatomicFW_SPL/
>> conda create -n pfw python=3.9
>> conda activate pfw
>> pip install -r requirements.txt
```

### Run the code

Once the dependencies installed and the environment activated, you can run reconstructions with PFW and compare them with the provided baselines (FISTA, Vanilla FW, Fully-Corrective FW). A testcase on a simulated sparse problem is showcased in the code ```example.py```.

In order to reproduce the plots of the paper, you should run the code ```plot_comparison_cs.py```. This code will create 6 subplots similar to the ones provided in the paper (warning: it can be relatively long to run).

The exact results of the paper are stored in the Pickle file ```results.p``` and can be plotted using ```plot_saved_results.py```.
