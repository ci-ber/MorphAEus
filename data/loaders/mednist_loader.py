from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import AddChannelIfNeeded, AssertChannelFirst, ReadImage, To01, MinMax
from transforms.preprocessing import Pad, Slice


class MedNISTDataset(DefaultDataset):
    def __init__(self, data_dir, file_type='', label_dir=None, target_size=(64, 64), test=False):
        super(MedNISTDataset, self).__init__(data_dir, file_type, label_dir, target_size, test)

    def get_label(self, idx):
        path_name = self.files[idx]
        if 'CXR' in path_name:
            return 0
        elif 'Abdomen' in path_name:
            return 1
        elif 'Head' in path_name:
            return 2
        elif 'Hand' in path_name:
            return 3
        elif 'Breast' in path_name:
            return 4
        else:
            return 0

    def get_image_transform(self):
        TPIL = transforms.ToPILImage()
        TT = transforms.ToTensor()
        RES = transforms.Resize(self.target_size)
        default_t = transforms.Compose([ReadImage(), To01(), TPIL, RES, TT])
        return default_t