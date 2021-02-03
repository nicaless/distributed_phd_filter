# README

Code for "Resilience in multi-robot multi-target tracking with unknown number of targets through reconfiguration" which includes Python implementations the distributed multi-robot PHD filter and resilient reconfiguration methodology described in the paper.

## Required Packages 

Assumes installation of Python3 and valid [MOSEK](https://www.mosek.com/) license.
```
cvxopt==1.2.5
decorator==4.4.2
llvmlite==0.35.0
Mosek==9.2.36
networkx==2.5
numba==0.52.0
numpy==1.19.5
pandas==1.1.5
PICOS==2.1.2
python-dateutil==2.8.1
pytz==2021.1
scipy==1.5.4
six==1.15.0
```

## Simple Usage

To run a simulation trial for 5 robot trackers execute:

````
python run_sim.py 5 [SAVE FOLDER]
````
