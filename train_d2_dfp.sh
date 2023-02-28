#!/bin/bash
#PBS -k o
#PBS -l nodes=nano8:gpus=1,walltime=120:00:00
#PBS -M ikeda2@illinois.edu
#PBS -m abe
#PBS -N D2-pytorch
#PBS -j oe

source activate nano8
cd /home/xdeng12/lustre/DFP-pytorch
python main.py -c ./scenarios/D2_navigation.cfg -n d2-baseline_labels+depth -a 8 -s --depth --labels
