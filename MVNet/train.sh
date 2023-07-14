# Model Training

# Stage1: Training Warm-up for model
#CUDA_VISIBLE_DEVICES=0,1 python main.py --backbone deeplab50 --training --baseline-mode --gpus 2 --lr-start 2e-4 --lr-strategy plateau_08 --num-epochs 150 --batch-size 2 --stm-queue-size 3 --sample-rate 3 --savedir save/training_base_deeplab

# Stage2: Training with memory
CUDA_VISIBLE_DEVICES=0,1 python main.py --backbone deeplab50 --weights save/training_base_deeplab/model_best.pth --training --gpus 2 --lr-start 2e-4 --lr-strategy plateau_08 --num-epochs 200 --batch-size 2 --stm-queue-size 3 --sample-rate 3 --savedir save/training_msa_deeplab