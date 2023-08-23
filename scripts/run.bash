#!bin/bash
. "/localdata/code/etc/profile.d/conda.sh"
export PYTHONPATH=/lhome/mr095415/linux/github/bispectrum
source activate py3 
echo "tracer" $1
echo "stat" $2
echo "reduce" $3
echo "template" $4
echo "use diag" $5
python run_mcmc.py $1 $2 $3 $4 $5
