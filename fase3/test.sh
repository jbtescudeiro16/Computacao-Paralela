#!/bin/bash
#SBATCH --time=5:00
#SBATCH --partition=cpar
#SBATCH --constraint=k20
#SBATCH --ntasks=1


time nvprof ./bin/MDCuda < inputdata.txt