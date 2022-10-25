from torchvision.datasets import SVHN
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = True

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = SVHN(
            root='/mnt/inspurfs/user-fs/wuxuefeng/Oliver/FPCC-F/svhn',
            split='train', download=True, transform=transform_train)

        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=4, pin_memory=pin_memory
        )

        testset = SVHN(
            root='/mnt/inspurfs/user-fs/wuxuefeng/Oliver/FPCC-F/svhn',
            split='test', download=True, transform=transform_test)
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=True,
            num_workers=4, pin_memory=pin_memory)