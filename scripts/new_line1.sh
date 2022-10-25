#!/usr/bin/env bash
 
echo "开始"

python ../pruning_cifar10.py --arch resnet56 --rate_dist 0.55 --pretrain_path /home/test6103/Oliver/YeQi/FPCC-F/Baseline/resnet_56.pt --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet56/lamda-0.55
wait
python ../pruning_cifar10.py --arch resnet56 --rate_dist 0.6 --pretrain_path /home/test6103/Oliver/YeQi/FPCC-F/Baseline/resnet_56.pt --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet56/lamda-0.6
wait

echo "结束"
