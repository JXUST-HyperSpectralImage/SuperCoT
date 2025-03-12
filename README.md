# SuperCoT


python run_conmim_pretraining.py \
            --mlp_dim 256 --patch_size 11 --temp 0.25 --mask_ratio 0.75 \
            --num_superpixel 450 --gamma 0.5 --batch_size 64 --lr 0.0003 --warmup_epochs 10 --epochs 200 \
            --dataset IndianPines --clip_grad 1.0 --drop_path 0 --layer_scale_init_value 1e-5 \
            --mask_type 'random_mps32' --output_dir ./output/pretrain \
            --save_ckpt_freq 200

python run_class_finetuning.py \
                --finetune  \
                --output_dir  \
                --dataset IndianPines --batch_size 64 --patch_size 11 --lr 0.03 --update_freq 1 \
                --warmup_epochs 10 --epochs 100 --layer_decay 0.65 --drop_path 0.1 \
                --run 10 --weight_decay 0.05 --load_data 0.10
