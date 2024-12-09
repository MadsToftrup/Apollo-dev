torchrun --standalone --nproc_per_node 8 main_pretrain.py \
    --model_config configs/llama_7b.json \
    --eval_every 1000 \
    --dtype bfloat16 \
    --batch_size 4 \
    --total_batch_size 512 \
    --lr 0.0025 \
    --warmup_steps 15000 \
    --num_training_steps 150000 \
    --optimizer adamw \
    --weight_decay 0 \
    --project apollo_test \
    --name apollo_test_adam \
    --save_dir ./ckpts/adam


