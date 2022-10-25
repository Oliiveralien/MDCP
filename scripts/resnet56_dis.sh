#!/usr/bin/env bash
 
echo "开始"

python ../pruning_cifar10.py  --dist_type seuclidean  --save_path ../logs/resnet56_dis/seuclidean
wait
python ../pruning_cifar10.py  --dist_type cityblock  --save_path ../logs/resnet56_dis/cityblock
wait
python ../pruning_cifar10.py  --dist_type cosine  --save_path ../logs/resnet56_dis/cosine
wait
python ../pruning_cifar10.py  --dist_type euclidean  --save_path ../logs/resnet56_dis/euclidean
wait
python ../pruning_cifar10.py  --dist_type correlation  --save_path ../logs/resnet56_dis/correlation

echo "结束"
