num_gpu=4

model_name=bloomz
k="176b"
torchrun --nproc_per_node=$num_gpu experiment_176b.py \
    --model_name $model_name \
    --model_size $k \
    --pretrained False
