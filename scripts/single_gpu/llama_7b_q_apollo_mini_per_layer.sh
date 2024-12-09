# LLaMA-7B, single GPU, 12GB
num_rank=1
scale_type=tensor
proj_type=random
apollo_scale=128

torchrun --standalone --nproc_per_node 1 main_pretrain.py \
    --model_config configs/llama_7b.json \
    --eval_every 1000 \
    --dtype bfloat16 \
    --batch_size 1 \
    --total_batch_size 1 \
    --lr 0.01 \
    --warmup_steps 15000 \
    --num_training_steps 150000 \
    --optimizer q_apollo_per_layer \
    --weight_quant \
    --weight_group_size 128 \
    --stochastic_round \
    --apollo_scale ${apollo_scale} \
    --rank ${num_rank} \
    --scale_type ${scale_type} \
    --proj ${proj_type} \
    --update_proj_gap 200 \
    --weight_decay 0 \
    --single_gpu \
    --project apollo_test \
    --name apollo_test_q_apollo_mini_7b_per_layer \
    --save_dir ./ckpts/Appollo_7b_s${seq_length}_scale${apollo_scale}_rank${num_rank}_proj${proj_type}_type${scale_type}


