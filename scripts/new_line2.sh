echo "开始"

python ../pruning_cifar10.py --arch resnet110 --rate_norm 0.8 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/k-0.2
wait
python ../pruning_cifar10.py --arch resnet110 --rate_norm 0.75 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/k-0.25
wait
python ../pruning_cifar10.py --arch resnet110 --rate_norm 0.7 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/k-0.3
wait
python ../pruning_cifar10.py --arch resnet110 --rate_norm 0.65 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/k-0.35
wait
python ../pruning_cifar10.py --arch resnet110 --rate_norm 0.6 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/k-0.4
wait

python ../pruning_cifar10.py --arch resnet110 --rate_dist 0.4 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/lamda-0.4
wait
python ../pruning_cifar10.py --arch resnet110 --rate_dist 0.45 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/lamda-0.45
wait
python ../pruning_cifar10.py --arch resnet110 --rate_dist 0.5 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/lamda-0.5
wait
python ../pruning_cifar10.py --arch resnet110 --rate_dist 0.55 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/lamda-0.55
wait
python ../pruning_cifar10.py --arch resnet110 --rate_dist 0.6 --save_path /home/test6103/Oliver/YeQi/FPCC-F/logs/Kpic/cifar10-resnet110/lamda-0.6

echo "结束"