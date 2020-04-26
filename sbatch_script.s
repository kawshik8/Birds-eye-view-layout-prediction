#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out

source ../../CV_project/env/bin/activate
module load python3/intel/3.5.3

python3 src/main.py --batch-size 4 --report-interval 1000 --exp-name "default" --results-dir ./results/ --data-dir ./data/ --image-folder /scratch/kk4161/DL_project/data --finetune-obj "det_encoder" --blobs-strategy "encoder_fused" --synthesizer-nlayers 2 --network-base resnet50 --finetune_learning_rate 0.001 --finetune-total-iters 1000000
