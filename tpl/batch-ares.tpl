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
cd FourCastNet
python inference.py \
       --config=afno_backbone \
       --run_num=0 \
       --data_path '/net/pr2/projects/plgrid/plggorheuro/fourcast/data/{{= filename }}.h5' \
       --weights '/net/pr2/projects/plgrid/plggorheuro/fourcast/FCN_weights_v0/backbone.ckpt' \
       --override_dir '{{= rootpath }}/{{= datapath }}/'

cd ..
cp {{= datapath }}/autoregressive_predictions_z500_vis.h5 bounce/prediction-{{= year }}-{{= month }}-{{= day }}.h5
rm -rf {{= datapath }}
rm -f out_of_sample/data.h5
