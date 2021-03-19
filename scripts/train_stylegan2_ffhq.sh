python tools/train_gan.py \
    --job stylegan2_ffhq-1024 \
    --data_path /dataset/ffhq/images-wrap --resume \
    --channel_multiplier 2 --resolution 1024 \
    --lr 0.002 --epochs 357 --batch_size 4 -j 2 \
    --fid_n_sample 50000 --fid_batch_size 8 --inception_path assets/inceptions/inception_ffhq_res1024_50k.pkl
