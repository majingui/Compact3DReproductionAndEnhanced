# 训练

# 真实数据集
CUDA_VISIBLE_DEVICES=2 python train.py -s /data2/mjg/data/dataset/mipnerf360/bicycle --eval --comp --store_npz
# 合成数据集
CUDA_VISIBLE_DEVICES=6 python train.py -s /data2/mjg/data/dataset/nerf_synthetic/chair  --eval --max_hashmap 16 --lambda_mask 4e-3 --mask_lr 1e-3 --net_lr 1e-3 --net_lr_step 25000


# 渲染
CUDA_VISIBLE_DEVICES=3 python render.py --iteration 30000 -s /data2/mjg/data/dataset/mipnerf360/kitchen -m /data2/mjg/work_space/3dgs/Compact-3DGS/output/e8f8d0c1-f  --quiet --eval --skip_train

# 评估
CUDA_VISIBLE_DEVICES=5 python metrics.py -m /data2/mjg/work_space/3dgs/Compact-3DGS/output/e8f8d0c1-f