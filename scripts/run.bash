#!bin/bash

. "/home/mr095415/miniconda3/etc/profile.d/conda.sh"
conda activate py3p8

export PYTHONPATH=/home/mr095415/bispectrum

# 
mock=$1
temp=$mock
stat=$2

mpirun -np 11 python mcmc_bispectrum.py --output_path ../mcmc_jan31/ --stat $stat --mock $mock --temp $temp -v
