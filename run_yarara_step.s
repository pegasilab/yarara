#!/bin/bash

#SBATCH --job-name=yarara_step
#SBATCH --output=SLURM/yarara_step_%j.out
#SBATCH --error=SLURM/yarara_step_%j.err
#SBATCH --ntasks=1
#SBATCH -p dace
#SBATCH --mem=40G

# run the simulation

python yarara_launcher_lesta.py -s $1 -i $2 -a 1 -l $3 -f 0 -r 0 -t $4 -D $5 -p $6
echo 'SUCESS COMPUTING RV FOR ALL LINES'



