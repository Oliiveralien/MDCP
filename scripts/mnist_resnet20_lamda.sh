#!/usr/bin/env bash
 
echo "开始"

python ../pruning_mnist.py --arch resnet20 --rate_dist 0.45 --save_path ../logs/resnet20_mnist/lamda-0.45
wait
python ../pruning_mnist.py --arch resnet20 --rate_dist 0.6 --save_path ../logs/resnet20_mnist/lamda-0.6
wait
python ../pruning_mnist.py --arch resnet20 --rate_dist 0.7 --save_path ../logs/resnet20_mnist/lamda-0.7
wait
python ../pruning_mnist.py --arch resnet20 --rate_dist 0.8 --save_path ../logs/resnet20_mnist/lamda-0.8


echo "结束"
