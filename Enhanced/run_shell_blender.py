import os

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['chair', 'drums', 'ficus', 'hotdog']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={6} python train.py -s /data2/mjg/data/dataset/nerf_synthetic/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 4 --iterations 30_000 -m outputs/nerf_synthetic_featpredict_fix/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)

for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['lego', 'materials', 'mic', 'ship']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={7} python train.py -s /data2/mjg/data/dataset/nerf_synthetic/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 4 --iterations 30_000 -m outputs/nerf_synthetic_featpredict_fix/{scene}/{lmbda} --lmbda {lmbda}'
        os.system(one_cmd)
