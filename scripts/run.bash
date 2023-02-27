#!bin/bash

. "/home/mr095415/miniconda3/etc/profile.d/conda.sh"
conda activate py3p8

export PYTHONPATH=/home/mr095415/bispectrum

# 
mock=$1
temp=$mock
gal=$2
stat=$3

mpirun -np 11 python mcmc_bispectrum.py --output_path ../mcmc_feb27/ --stat $stat --mock $mock --temp $temp --gal $gal -v
#mpirun -np 11 python mcmc_bispectrum_joint.py --output_path ../mcmc_feb27/ --stat $stat --mock $mock --temp $temp --gal $gal -v
#mpirun -np 11 python mcmc_bispectrum_nobaojoint.py --output_path ../mcmc_feb27/ --stat $stat --mock $mock --temp $temp --gal $gal -v
