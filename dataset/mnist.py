from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = True

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
# /mnt/inspurfs/user-fs/wuxuefeng/Oliver/FPCC-F/mnist
        trainset = MNIST(
            root='../mnist',
            train=True, download=True, transform=transform_train)

        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=4, pin_memory=pin_memory
        )

        testset = MNIST(
            root='../mnist',
            train=False, download=True, transform=transform_test)
            # train=False, download=True, transform=transforms.ToTensor())
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=True,
            num_workers=4, pin_memory=pin_memory)