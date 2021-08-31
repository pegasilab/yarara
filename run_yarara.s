#!/bin/bash

#SBATCH --job-name=yarara
#SBATCH --output=SLURM/yarara_%j.out
#SBATCH --error=SLURM/yarara_%j.err
#SBATCH --ntasks=1
#SBATCH -p dace
#SBATCH --mem=50G

# run the simulation

python yarara_launcher_lesta.py -s $1 -i $2 -a 1 -l $3 -f 0 -r 0 -t yarara -D $4 -p $5
echo 'SUCESS COMPUTING RV FOR ALL LINES'



