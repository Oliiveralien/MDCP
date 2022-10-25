#!/usr/bin/env bash
 
echo "开始"

python ../pruning_mnist_1.py --arch resnet32 --rate_dist 0.45 --save_path ../logs/resnet32_mnist/lamda-0.45
wait
python ../pruning_mnist_1.py --arch resnet32 --rate_dist 0.6 --save_path ../logs/resnet32_mnist/lamda-0.6
wait
python ../pruning_mnist_1.py --arch resnet32 --rate_dist 0.7 --save_path ../logs/resnet32_mnist/lamda-0.7
wait
python ../pruning_mnist_1.py --arch resnet32 --rate_dist 0.8 --save_path ../logs/resnet32_mnist/lamda-0.8


echo "结束"
