import torchvision.transforms as transforms
import pytorch_lightning as pl
import hub


class CelebA(pl.LightningDataModule):
    """
    Celebrity dataset:
        Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In Proceedings
        of International Conference on Computer Vision (ICCV), December 2015
    """
    def __init__(self, args):
        akeys = args.keys()
        self.target_size = args['target_size'] if 'target_size' in akeys else (64, 64)
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 2
        TPIL = transforms.ToPILImage()
        TT = transforms.ToTensor()
        RES = transforms.Resize(self.target_size)
        Gray = transforms.Grayscale()

        # Transforms object for testset with NO augmentation
        self.trans = {'images':  transforms.Compose([TPIL, Gray, RES, TT])}

    def train_dataloader(self):
        trainset = hub.load("hub://activeloop/celeb-a-train")
        dataloader = trainset.pytorch(num_workers=self.num_workers,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      transform=self.trans)
        return dataloader

    def val_dataloader(self):
        trainset = hub.load("hub://activeloop/celeb-a-val")
        dataloader = trainset.pytorch(num_workers=self.num_workers,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      transform=self.trans)
        return dataloader

    def test_dataloader(self):
        trainset = hub.load("hub://activeloop/celeb-a-test")
        dataloader = trainset.pytorch(num_workers=self.num_workers,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      transform=self.trans)
        return dataloader
