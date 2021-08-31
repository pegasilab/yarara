#!/bin/bash

#SBATCH --job-name=rassine
#SBATCH --output=SLURM/rassine_%j.out
#SBATCH --error=SLURM/rassine_%j.err
#SBATCH --ntasks=6
#SBATCH -N 1
#SBATCH -p dace
#SBATCH --mem=40G

# run the simulation

python yarara_launcher_lesta.py -s $1 -i $2 -a 1 -l $3 -f 0 -r $4 -t rassine -k $5 -D $6 -c $7
echo 'SUCESS COMPUTING RV FOR ALL LINES'



