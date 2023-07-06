
for m in bloomz bloom
do
for k in 7b1
do
    echo $m
    python -u pre_experiment.py \
        --model_name $m \
        --model_size $k \
        --en_prompt True
done
done
