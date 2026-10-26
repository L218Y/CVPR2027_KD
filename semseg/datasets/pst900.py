import os
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from typing import Tuple


class PST900(Dataset):
    """PST900 RGB-Thermal segmentation (5 classes incl. background).

    Labels are palette-index PNGs with values 0..4.  The official test split is
    exposed as ``official`` and must only be touched once per model.
    """
    CLASSES = ['background', 'fire extinguisher', 'backpack', 'hand drill', 'survivor']
    PALETTE = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])

    def __init__(self, root: str = 'data/PST900', split: str = 'train', transform=None,
                 modals=['img', 'thermal'], case=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'official']
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = self._get_file_names(split)
        if not self.files:
            raise Exception(f"No images found in {root} ({split})")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])
        rgb = os.path.join(self.root, 'rgb', item_name + '.png')
        x1 = os.path.join(self.root, 'thermal', item_name + '.png')
        lbl_path = os.path.join(self.root, 'labels', item_name + '.png')
        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        if 'thermal' in self.modals:
            sample['thermal'] = self._open_img(x1)
        label = io.read_image(lbl_path)[0, ...].unsqueeze(0)
        sample['mask'] = label
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label) -> Tensor:
        label = torch.from_numpy(label)
        return torch.where(label == 255, torch.full_like(label, self.ignore_label), label)

    def _get_file_names(self, split_name):
        source = os.path.join(self.root, {'train': 'train.txt', 'val': 'test.txt',
                                          'official': 'official.txt'}[split_name])
        names = []
        with open(source) as f:
            for item in f:
                name = item.strip().split(' ')[0] if item.strip() else ''
                if name:
                    names.append(name)
        return names
