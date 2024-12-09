# LLaMA-1B, APOLLO, 8 A100, 1 Node
num_rank=256
scale_type=channel
proj_type=random
apollo_scale=1

torchrun --standalone --nproc_per_node 8 main_pretrain.py \
    --model_config configs/llama_1b.json \
    --eval_every 1000 \
    --dtype bfloat16 \
    --batch_size 64 \
    --total_batch_size 512 \
    --lr 0.01 \
    --warmup_steps 10000 \
    --num_training_steps 100000 \
    --optimizer apollo \
    --apollo_scale ${apollo_scale} \
    --rank ${num_rank} \
    --scale_type ${scale_type} \
    --proj ${proj_type} \
    --update_proj_gap 200 \
    --weight_decay 0 \
    --project apollo_test \
    --name apollo_test_apollo_1b \
    --save_dir ./ckpts/Appollo_7b_scale${apollo_scale}_rank${num_rank}_proj${proj_type}_type${scale_type}


