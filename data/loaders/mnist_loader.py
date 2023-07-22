import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class MNISTLoader(pl.LightningDataModule):
    def __init__(self, args):
        akeys = args.keys()
        self.target_size = args['target_size'] if 'target_size' in akeys else (64, 64)
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 2

        TT = transforms.ToTensor()
        RES = transforms.Resize(self.target_size)
        # Norm = transforms.Normalize((0.1307,), (0.3081,))
        # Transforms object for testset with NO augmentation
        self.transform_no_aug = transforms.Compose([RES, TT])

    def train_dataloader(self):
        trainset = tv.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform_no_aug)
        dataloader = DataLoader(trainset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def val_dataloader(self):
        trainset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform_no_aug)
        dataloader = DataLoader(trainset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
