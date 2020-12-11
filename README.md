# CBLBs: Configurable (Bio)Logic Blocks

This repository is supplementing the paper Field-programmable biological circuits and configurable (bio)logic blocks for distributed biological computing [1]. 

## Folders
The project is organized in the following folders
* [```cblb```](/cblb/): deterministic and stochastic models of a CBLB and its modules.
* [```robustness```](/robustness/): analysis of robustness.
* [```latch```](/latch/): model of a multicellular latch, i.e. toggle switch, as described in [2].
* [```mux```](/mux/): model of a multicellular multiplexer and derivation of its kinetic parameters using the transfer-function models and values from [3].


## Main files
The main files that can be used to reproduce the results reported in the paper [1] are as follows
* [```cblb/models.py```](/cblb/models.py): the implementation of deterministic and stochastic models.
* [```cblb/run_ode_model_clb.py```](/cblb/run_ode_model_clb.py): deterministic simulation of a CBLB.
* [```cblb/run_ssa_model_clb.py```](/cblb/run_ssa_model_clb.py): stochastic simulation of a CBLB.
* [```cblb/run_ODE_AND_OR_NOT.py```](/cblb/run_ODE_AND_OR_NOT.py): deterministic simulation of a CBLB configured as AND, OR and NOT gate.
* [```robustness/run_solver.py```](/robustness/run_solver.py): perform the exploration of viable parameter space as described in [4].
* [```robustness/run_solver_population.py```](/robustness/run_solver_population.py): perform the exploration of viable parameter space [4] investigating also viable population ratios.
* [```robustness/robustness_analysis.py```](/robustness/robustness_analysis.py): visualize and analyse the results of the obtained viable parameter space.
* [```robustness/robustness_analysis_population.py```](/robustness/robustness_analysis_population.py): visualize the results of the analysis of different population ratio effects.

## Requirements
You should install the following libraries before running the code 
* `DEAP`
* `SciPy`
* `NumPy`
* `seaborn`
* `matplotlib`
* `pandas`
* `sklearn`

## How to cite this work
Please cite this work as

Moškon M, Pušnik Ž, Zimic N, Mraz M. Field-programmable biological circuits and configurable (bio)logic blocks for distributed biological computing. Computers in Biology and Medicine. 2021; 128: 104109, https://doi.org/10.1016/j.compbiomed.2020.104109.



## References:

[1] Moškon M, Pušnik Ž, Zimic N, Mraz M. Field-programmable biological circuits and configurable (bio)logic blocks for distributed biological computing. Computers in Biology and Medicine. 2021; 128: 104109, https://doi.org/10.1016/j.compbiomed.2020.104109.


[2] Urrios A, Macia J, Manzoni R, Conde N, Bonforti A, de Nadal E, Posas F, Solé R. A synthetic multicellular memory device. ACS synthetic biology. 2016 Aug 19;5(8):862-73.

[3] Macia J, Manzoni R, Conde N, Urrios A, de Nadal E, Solé R, Posas F. Implementation of complex biological logic circuits using spatially distributed multicellular consortia. PLoS computational biology. 2016 Feb 1;12(2):e1004685.

[4] Pušnik Ž, Mraz M, Zimic N, Moškon M. Computational analysis of viable parameter regions in models of synthetic biological systems. Journal of biological engineering. 2019 Dec 1;13(1):75.