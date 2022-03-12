export PYTHONPATH=$PYTHONPATH:$PWD/../..

python run.py --n-exp 20 --n-jobs 10 --use-curriculum --use-boosting
python run.py --n-exp 20 --n-jobs 10 --use-boosting
python run.py --n-exp 20 --n-jobs 10 --use-curriculum
python run.py --n-exp 20 --n-jobs 10
python visualize_results.py
