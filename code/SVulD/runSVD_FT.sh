#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --job-name=M
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH -p long-disi
#SBATCH -N 1
#SBATCH --time=40:00:00 



for learning_rate in 1e-5 2e-5
do
    for max_grad_norm in 0.6 0.8 1.0
    do 
        for weight_decay in 0.1 0.4 0.2
        do
            for adam_epsilon in 2e-8 
            do 
                python3 run_add_FT.py \
                    --output_dir saved_models/gn_${max_grad_norm}_wd_${weight_decay}_ad_${adam_epsilon}_lr_${learning_rate} \
                    --model_name_or_path microsoft/unixcoder-base-nine \
                    --do_train \
                    --do_test \
                    --train_data_file ../../dataset/SVulDData/train.jsonl \
                    --eval_data_file ../../dataset/SVulDData/valid.jsonl \
                    --test_data_file ../../dataset/SVulDData/test.jsonl \
                    --num_train_epochs 20 \
                    --block_size 400 \
                    --train_batch_size 32 \
                    --eval_batch_size 32 \
                    --learning_rate $learning_rate \
                    --max_grad_norm $max_grad_norm \
                    --seed 99 \
                    --weight_decay $weight_decay \
                    --adam_epsilon $adam_epsilon \
                    --r_drop
            done
        done
    done
done


