export PYTHONPATH=$PYTHONPATH:$PWD/..

python train_lspi_linear.py --n-jobs 10 --n-exp 100
python train_dqn_linear.py --n-jobs 10 --n-exp 100
python train_nonlinear.py --n-jobs 10 --n-exp 100
python visualize_results.py
