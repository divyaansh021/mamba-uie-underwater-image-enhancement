import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp"])


def load_img(filepath):
    return Image.open(filepath).convert('RGB')


def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    # ✅ Paper-accurate: uniformly resize every image to 256×256
    ip = patch_size
    img_in = img_in.resize((ip, ip), resample=Image.BICUBIC)
    img_tar = img_tar.resize((ip, ip), resample=Image.BICUBIC)
    info_patch = {'ix': 0, 'iy': 0, 'ip': ip, 'tx': 0, 'ty': 0, 'tp': ip}
    return img_in, img_tar, info_patch


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True
    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True
    return img_in, img_tar, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, label_dir, patch_size, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.data_filenames = sorted([join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)])
        self.label_filenames = sorted([join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)])
        self.patch_size = patch_size
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.label_filenames[index])
        input = load_img(self.data_filenames[index])
        _, file = self.label_filenames[index].rsplit("/", 1)
        input, target, _ = get_patch(input, target, self.patch_size)

        if self.data_augmentation:
            input, target, _ = augment(input, target)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, file

    def __len__(self):
        return len(self.label_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.data_filenames = sorted([join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)])
        self.label_filenames = sorted([join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)])
        self.transform = transform

    def __getitem__(self, index):
        target = load_img(self.label_filenames[index])
        input = load_img(self.data_filenames[index])
        _, file = self.label_filenames[index].rsplit("/", 1)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, file

    def __len__(self):
        return len(self.label_filenames)
