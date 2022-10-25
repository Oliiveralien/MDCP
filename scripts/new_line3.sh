#!/usr/bin/env bash
 
echo "开始"
python ../pruning_cifar10.py --rate_norm 0.8 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet56/0.2
wait
python ../pruning_cifar10.py --rate_norm 0.75 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet56/0.25
wait
python ../pruning_cifar10.py --rate_norm 0.7 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet56/0.3
wait
python ../pruning_cifar10.py --rate_norm 0.65 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet56/0.35
wait
python ../pruning_cifar10.py --rate_norm 0.6 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet56/0.4

echo "结束"
