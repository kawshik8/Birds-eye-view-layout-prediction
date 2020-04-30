#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:k80:1
#SBATCH --time=30:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out

source ../../CV_project/env/bin/activate
module load python3/intel/3.5.3

########## IMAGE SSL
#python3 src/main.py --num-negatives 40000 --batch-size 64 --report-interval 250 --exp-name "pirl_r18" --results-dir ./results/ --data-dir ./data/ --image-folder /scratch/kk4161/DL_project/data --image-pretrain-obj "pirl_nce_loss" --network-base resnet18 --pretrain-learning-rate 0.01 --pretrain-total-iters 100000

############# SUPERVISED
#python3 src/main.py --batch-size 8 --report-interval 250 --exp-name "road_map_cvar512_bce" --detect-objects 0 --gen-road-map 1 --results-dir ./results/ --data-dir ./data/ --image-folder /scratch/kk4161/DL_project/data --finetune-obj "var_encoder" --blobs-strategy "encoder_fused" --synthesizer-nlayers 2 --network-base resnet18 --finetune-learning-rate 0.01 --finetune-total-iters 200000 --latent-dim 512

python3 src/main.py --batch-size 8 --report-interval 250 --exp-name "road_map_det_refineviews_bce" --detect-objects 0 --gen-road-map 1 --results-dir ./results/ --data-dir ./data/ --image-folder /scratch/kk4161/DL_project/data --finetune-obj "det_encoder" --blobs-strategy "encoder_fused" --synthesizer-nlayers 2 --network-base resnet18 --finetune-learning-rate 0.005 --finetune-total-iters 200000
