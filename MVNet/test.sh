# Model Inference

# Inference for two-stream baseline model
#python main.py --eval-mode --split-mode test --backbone deeplab50 --weights save/training_base_deeplab/model_best.pth --stm-queue-size 3 --sample-rate 3 --savedir save/eval_base_deeplab --baseline-mode

# Inference for the MVNet model
python main.py --eval-mode --split-mode test --backbone deeplab50 --weights save/training_msa_deeplab/model_best.pth --stm-queue-size 3 --sample-rate 3 --savedir save/eval_msa_deeplab
