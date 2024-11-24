import os

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={3} python train.py -s /data2/mjg/data/dataset/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360_featpredict_fix_no_warm/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)

#
# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['bicycle', 'garden',]):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s /data2/mjg/data/dataset/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360_featpredict_fix/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['stump','room']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s /data2/mjg/data/dataset/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360_featpredict_fix/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['counter', 'kitchen']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={2} python train.py -s /data2/mjg/data/dataset/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360_featpredict_fix/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)
#
for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['treehill']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={4} python train.py -s /data2/mjg/data/dataset/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360_featpredict_fix/{scene}/{lmbda} --lmbda {lmbda}'
        os.system(one_cmd)
# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['stump']):
#         one_cmd = f'CUDA_VISIBLE_DEVICES={2} python train.py -s /data2/mjg/data/dataset/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360_featpredict_fix_no_warm/{scene}/{lmbda} --lmbda {lmbda}'
#         os.system(one_cmd)