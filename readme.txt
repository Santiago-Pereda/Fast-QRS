This repository contains the following matlab m-files for the replication of the simulations in "Fast Algorithms for Quantile Regression with Selection" Pereda-Fernández (2025)

The article can be accessed at https://www.degruyterbrill.com/document/doi/10.1515/jem-2024-0022/html

The folder "Matlab files" contains all the functions that are needed to compute the Quantile Regression with Selection estimator. In particular, it includes the estimator without any time-saving algorithm (qrs.m), Algorithm 1 (rqrtau_fast.m), Algorithm 2 (rqr_fast.m), Algorithm 3 (qrs_fast_bt.m), and Algorithm 4 (qrs_fast_bt) for the bootstrap . In addition, it includes all the other necessary files that are used by these functions:

-rqrb0_fast.m

-rq.m

-rq_pen.m

-checks_rqr.m

-checkfn.m

In addition, it also contains some files that are used to obtain the simulations in the paper:

-simul_optim.m

-simul_bootstrap.m

-simul_precision.m

There is an additional file (example.m) that has a small example to showcase how the algorithms work.

A more detailed description of the input and output for each function can be found within each file