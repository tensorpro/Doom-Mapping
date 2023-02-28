#!/bin/bash
#PBS -k o
#PBS -l nodes=nano8:gpus=1,walltime=120:00:00
#PBS -M ikeda2@illinois.edu
#PBS -m abe
#PBS -N D4-pytorch
#PBS -j oe

source activate nano8
cd /home/xdeng12/lustre/DFP-pytorch
python depth+labels.py -c ./scenarios/D4_battle2.cfg -n d4_map-depth+labels
