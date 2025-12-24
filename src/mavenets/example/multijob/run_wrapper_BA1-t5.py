"""Example invocation of multijob hyperparameter scan function.

This script should be called with two arguments. The first indicates the index of a
given job, and the second is the total number of jobs that will be run simultaneously.
Furthermore, each job should be assigned to a different GPU via the CUDA_VISIBLE_DEVICES.
For example, if you want to run 4 different jobs, the you should launch:
    CUDA_VISIBLE_DEVICES=0 python run_example.py 0 4
    CUDA_VISIBLE_DEVICES=1 python run_example.py 1 4
    CUDA_VISIBLE_DEVICES=2 python run_example.py 2 4
    CUDA_VISIBLE_DEVICES=3 python run_example.py 3 4

This will run 4 different hyperparameter scans in parallel, each on its own GPU. The portion
of the hyperparameter space that each job will scan is determined by the two arguments.

"""
from sys import argv
from mavenets.example.multijob import run_mlp_BA1_nulltuner_t5_multijob as mm
# first is replica index, second is total number of replicas
replica = int(argv[1])
total_n_replicas = int(argv[2])
assert replica < total_n_replicas
mm.scan(replica,total_n_replicas)
