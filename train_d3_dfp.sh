#!/bin/bash
#PBS -k o
#PBS -l nodes=nano8:gpus=1,walltime=200:00:00
#PBS -M ikeda2@illinois.edu
#PBS -m abe
#PBS -N D3-pytorch
#PBS -j oe

source activate nano8
cd /home/xdeng12/lustre/DFP-pytorch
python depth+labels.py -c ./scenarios/D3_battle.cfg -n d3_map-depth+labels
