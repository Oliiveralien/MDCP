from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = True
        transform_train = transforms.Compose([

            transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225]),

        ])

        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225]),
        ])
        transform = transforms.Compose([

            transforms.RandomSizedCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225]),

        ])

        trainset = datasets.ImageFolder('/home/test9505/Imagenet/train', transform)
        # trainset = CIFAR10(
        #     root='/home/test9505/filter-pruning-geometric-median-master/geometric-median/data/cifar.python/cifar10',
        #     train=True, download=True, transform=transform_train)

        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=4, pin_memory=pin_memory
        )
        valdir = os.path.join('/home/test9505/Imagenet', 'val')

        testset = datasets.ImageFolder(valdir, transform)
        # testset = CIFAR10(
        #     root='/home/test9505/filter-pruning-geometric-median-master/geometric-median/data/cifar.python/cifar10',
        #     train=False, download=True, transform=transform_test)
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=True,
            num_workers=4, pin_memory=pin_memory)