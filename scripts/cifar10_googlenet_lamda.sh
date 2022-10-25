#!/usr/bin/env bash

echo "开始"

python ../googlenet_pruning_cifar10.py --arch googlenet --rate_dist 0.65 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-googlenet/lamda-0.65
wait
python ../googlenet_pruning_cifar10.py --arch googlenet --rate_dist 0.7 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-googlenet/lamda-0.7
wait
python ../googlenet_pruning_cifar10.py --arch googlenet --rate_dist 0.75 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-googlenet/lamda-0.75
wait
python ../googlenet_pruning_cifar10.py --arch googlenet --rate_dist 0.8 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-googlenet/lamda-0.8
wait
python ../googlenet_pruning_cifar10.py --arch googlenet --rate_dist 0.85 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-googlenet/lamda-0.85
wait
python ../googlenet_pruning_cifar10.py --arch googlenet --rate_dist 0.9 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-googlenet/lamda-0.9

echo "结束"