This folder contains the code necessary to run the experiments presented in the paper. We used Python 3.7.10 and
the packages listed in the **requirements.txt** file to run the experiments. The dependencies listed in this file
can be installed with pip

```shell script
pip install -r requirements.txt
```

The three experiments from the paper can be found in **fqi/car_on_hill**, **fqi/maze** and **linear_dqn**. Each
directory contains a script file **run_jobs.sh** that can be used to run the experiments. There is a **--n-jobs**
option for the scripts in the fqi directory. This option controls the number of workers used to run the experiments
and can be set higher if more cores are available on the machine executing the experiments. Note that the **fqi**
scripts take a significant amount of time to run, as e.g. the brute-force solution of the car-on-hill task take
a considerable amount of computation time.

Finally, each directory contains a script **visualize_results.py** that was used to create the plots in the paper and
can be run via

```shell script
python visualize_results.py
```