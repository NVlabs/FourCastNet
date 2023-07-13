#!/bin/bash

#SBATCH --job-name=fourcast
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --time=01:00:00
#SBATCH --account=plgmeteoml2-gpu
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -N 1
#SBATCH --output=/net/ascratch/people/plgorhid/fourcast-output/fourcast-slurm.out
#SBATCH --error=/net/ascratch/people/plgorhid/fourcast-output/fourcast-slurm.err

module load python

cd {{= rootpath }}
python src/inference.py \
       --yaml_config '/net/pr2/projects/plgrid/plggorheuro/fourcast/src/config/AFNO.yaml' \
       --config=afno_backbone \
       --run_num=0 \
       --data_path '/net/pr2/projects/plgrid/plggorheuro/fourcast/data/test.h5' \
       --weights '/net/pr2/projects/plgrid/plggorheuro/fourcast/model-weights/backbone.ckpt' \
       --override_dir '/net/ascratch/people/plgorhid/fourcast-output/'
