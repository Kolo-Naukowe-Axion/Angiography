import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def _files_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {path.stem: path for path in sorted(directory.iterdir()) if path.is_file()}


def _pair_image_mask_paths(images_dir: Path, masks_dir: Path) -> list[tuple[str, str]]:
    images = _files_by_stem(images_dir)
    masks = _files_by_stem(masks_dir)
    paired: list[tuple[str, str]] = []
    for stem in sorted(images.keys()):
        mask_path = masks.get(stem)
        if mask_path is None:
            continue
        paired.append((str(images[stem]), str(mask_path)))
    return paired


def _triple_image_mask_feature_paths(images_dir: Path, masks_dir: Path, feature_dir: Path) -> list[tuple[str, str, str]]:
    images = _files_by_stem(images_dir)
    masks = _files_by_stem(masks_dir)
    features = _files_by_stem(feature_dir)
    triples: list[tuple[str, str, str]] = []
    for stem in sorted(images.keys()):
        mask_path = masks.get(stem)
        feature_path = features.get(stem)
        if mask_path is None or feature_path is None:
            continue
        triples.append((str(images[stem]), str(mask_path), str(feature_path)))
    return triples


class Branch1_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, test=False):
        super(Branch1_datasets, self)
        base_path = Path(path_Data)
        if train:
            self.data = _pair_image_mask_paths(base_path / "train" / "images", base_path / "train" / "masks")
            self.transformer = config.train_transformer
        else:
            if test:
                self.data = _pair_image_mask_paths(base_path / "test" / "images", base_path / "test" / "masks")
                self.transformer = config.test_transformer
            else:
                self.data = _pair_image_mask_paths(base_path / "val" / "images", base_path / "val" / "masks")
                self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


class Branch2_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, test=False):
        super(Branch2_datasets, self)
        base_path = Path(path_Data)
        if train:
            self.data = _triple_image_mask_feature_paths(
                base_path / "train" / "images",
                base_path / "train" / "masks",
                base_path / "train" / "feature",
            )
            self.transformer = config.train_transformer
        else:
            if test:
                self.data = _triple_image_mask_feature_paths(
                    base_path / "test" / "images",
                    base_path / "test" / "masks",
                    base_path / "test" / "feature",
                )
                self.transformer = config.test_transformer
            else:
                self.data = _triple_image_mask_feature_paths(
                    base_path / "val" / "images",
                    base_path / "val" / "masks",
                    base_path / "val" / "feature",
                )
                self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path, feature_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        feature = torch.load(feature_path, map_location='cpu')

        if self.transformer is not None:
            img, msk = self.transformer((img, msk))
        return img, msk, feature

    def __len__(self):
        return len(self.data)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
    
