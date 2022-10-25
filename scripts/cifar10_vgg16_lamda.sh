#!/usr/bin/env bash
 
echo "开始"

python ../vgg_pruning_cifar10.py --arch vgg16 --rate_dist 0.4 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-vgg16/lamda-0.4
wait
python ../vgg_pruning_cifar10.py --arch vgg16 --rate_dist 0.45 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-vgg16/lamda-0.45
wait
python ../vgg_pruning_cifar10.py --arch vgg16 --rate_dist 0.5 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-vgg16/lamda-0.5
wait
python ../vgg_pruning_cifar10.py --arch vgg16 --rate_dist 0.55 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-vgg16/lamda-0.55
wait
python ../vgg_pruning_cifar10.py --arch vgg16 --rate_dist 0.6 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-vgg16/lamda-0.6


echo "结束"
