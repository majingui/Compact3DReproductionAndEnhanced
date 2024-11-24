import os

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['truck', 'train']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={4} python train.py -s /data2/mjg/data/dataset/tanks_and_temples/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 40_000 -m outputs/tandt_gpu_4/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['truck', 'train']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={4} python train.py -s /data2/mjg/data/dataset/tanks_and_temples/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt_gpu_4/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['truck', 'train']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={2} python train.py -s /data2/mjg/data/dataset/tanks_and_temples/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['truck', 'train']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={4} python train.py -s /data2/mjg/data/dataset/tanks_and_temples/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 40_000 -m outputs/tandt_gpu_4/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)

for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['train','truck']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={4} python train.py -s /data2/mjg/data/dataset/tanks_and_temples/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tanks_and_temples_featpredict_fix/{scene}/{lmbda} --lmbda {lmbda}'
        os.system(one_cmd)
