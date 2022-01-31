# Polyatomic Frank-Wolfe for the LASSO

This repository provides the code to run and reproduce the results for the IEEE Signal Processing Letters paper "A Fast and Scalable Polyatomic Frank-Wolfe Algorithm for the LASSO", from Adrian Jarret, Julien Fageot and Matthieu Simeoni.

The implementation is based on the optimization package [Pycsou](https://github.com/matthieumeo/pycsou)

## Installation

The notebook requires Python 3.??? or greater. It is developed and tested on x86_64 systems running Linux.


### Dependencies

The notebook extra dependencies are listed in the file ``requirements.txt``.
It is recommended to install those extra dependencies in an Anaconda environment [Miniconda](https://conda.io/miniconda.html) or
[Anaconda](https://www.anaconda.com/download/#linux). 

```bash
>> conda create -n pfw python=3.9
>> conda activate pfw
>> pip install -r requirements.txt
```

### Run the code

Once the dependencies installed, you can run the companion notebook by executing the following commands: 

```bash
>> git clone https://github.com/AdriaJ/PolyatomicFW_SPL
>> cd PolyatomicFW_SPL/
>> conda activate pfw
```
