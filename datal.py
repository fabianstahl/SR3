import cv2
import numpy as np
import configparser
import pytorch_lightning as pl
import torch
import random

def normalize(x):
    return (x / 128) - 1

def denormalize(x):
    return (x * 128) + 128




def collate(batch, idx):
    x   = torch.tensor(np.array([sample['SR'] for sample in batch]))
    gt  = torch.tensor(np.array([sample['HR'] for sample in batch]))

    # [b, w, h, c] -> [b, c, h, w]
    x   = x.permute(0, 3, 1, 2)
    gt  = gt.permute(0, 3, 1, 2)

    return {'HR': gt, 'SR': x, 'Index': idx}



class SuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.window_size    = config.getint('WindowSize')
        self.gt_img         = cv2.imread(config.get('GroundTruthImagePath'))
        w, h, _             = self.gt_img.shape
        scale_factor        = config.getint('ScaleFactor')
        self.inp_img        = cv2.resize( cv2.resize(self.gt_img, (h // scale_factor, w // scale_factor)), (h, w))
        self.size           = self._calculate_size()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        vres, hres, _   = self.inp_img.shape
        h_samples       = hres // self.window_size
        v_samples       = vres // self.window_size
        h_start         = (idx % h_samples) * self.window_size
        v_start         = (idx // h_samples) * self.window_size
        x               = self.inp_img[v_start : v_start + self.window_size, h_start : h_start + self.window_size]
        gt              = self.gt_img[v_start : v_start + self.window_size, h_start : h_start + self.window_size]
        return {'x':x, 'gt':gt}

    def _calculate_size(self):
        hres, vres, _ = self.inp_img.shape
        return (hres // self.window_size) * (vres // self.window_size)




class AugmentableUltraDataset(torch.utils.data.Dataset):
    def __init__(self, config, augmentation = False):
        self.window_size    = config.getint('WindowSize')
        self.gt_img         = cv2.imread(config.get('GroundTruthImagePath'))
        w, h, _             = self.gt_img.shape
        scale_factor        = config.getint('ScaleFactor')
        self.inp_img        = cv2.resize( cv2.resize(self.gt_img, (h // scale_factor, w // scale_factor)), (h, w))
        self.max_offset     = config.getint('MaxOffset')
        self.size           = self._calculate_size()
        self.use_augmentation = augmentation # Warning: Needs to be switched on by hand to avoid wrong logging

    def load_with_augmentation(self, idx):

        # Calculate the required window size with padding in order to allow rotations and offseting up to "maxOffset"
        img_size_with_pad = 2 * self.max_offset + int(np.ceil(np.sqrt(2 * (self.window_size**2))))

        # Extract Image with padding
        vres, hres, _   = self.inp_img.shape
        h_samples   = hres // img_size_with_pad
        v_samples   = vres // img_size_with_pad
        h_start     = (idx % h_samples) * img_size_with_pad
        v_start     = (idx // h_samples) * img_size_with_pad
        image_inp   = self.inp_img[v_start : v_start + img_size_with_pad, h_start : h_start + img_size_with_pad]
        image_gt    = self.gt_img[v_start : v_start + img_size_with_pad, h_start : h_start + img_size_with_pad]

        # Perform random rotation
        rand_angle      = np.random.rand() * 360
        center          = (img_size_with_pad // 2, img_size_with_pad // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rand_angle, 1)
        image_inp       = cv2.warpAffine(image_inp, rotation_matrix, (image_inp.shape[1], image_inp.shape[0]))
        image_gt        = cv2.warpAffine(image_gt, rotation_matrix, (image_gt.shape[1], image_gt.shape[0]))

        # Perform random crop
        rand_offset = np.floor((np.random.rand(2) * 2 * self.max_offset) - self.max_offset).astype(int)
        h_start     = center[0] + rand_offset[0] - self.window_size // 2
        v_start     = center[1] + rand_offset[1] - self.window_size // 2

        image_inp   = image_inp[v_start : v_start + self.window_size, h_start : h_start + self.window_size]
        image_gt    = image_gt[v_start : v_start + self.window_size, h_start : h_start + self.window_size]

        return image_inp, image_gt

    def load_without_augmentation(self, idx):

        # Calculate the required window size with padding in order to allow rotations and offseting up to "maxOffset"
        img_size_with_pad = 2 * self.max_offset + int(np.ceil(np.sqrt(2 * (self.window_size**2))))

        # Extract centered image
        vres, hres, _   = self.inp_img.shape
        h_samples   = hres // img_size_with_pad
        v_samples   = vres // img_size_with_pad
        h_start     = (idx % h_samples) * img_size_with_pad + self.max_offset
        v_start     = (idx // h_samples) * img_size_with_pad + self.max_offset
        image_inp   = self.inp_img[v_start : v_start + self.window_size, h_start : h_start + self.window_size]
        image_gt    = self.gt_img[v_start : v_start + self.window_size, h_start : h_start + self.window_size]

        return image_inp, image_gt


    def set_augmentation(val):
        self.use_augmentation = val

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.use_augmentation:
            x, gt = self.load_with_augmentation(idx)
        else:
            x, gt = self.load_without_augmentation(idx)

        x   = torch.tensor(x).permute(2, 0, 1).float()
        gt  = torch.tensor(gt).permute(2, 0, 1).float()

        x = normalize(x)
        gt = normalize(gt)

        # [b, w, h, c] -> [b, c, h, w]
        #x   = x.permute(0, 3, 1, 2)
        #gt  = gt.permute(0, 3, 1, 2)

        return {'HR': gt, 'SR': x, 'Index': idx}

    def _calculate_size(self):
        hres, vres, _ = self.inp_img.shape
        img_size_with_pad = 2 * self.max_offset + int(np.ceil(np.sqrt(2 * (self.window_size**2))))
        return (hres // img_size_with_pad) * (vres // img_size_with_pad)



class UltraDataModule(pl.LightningDataModule):

    def __init__(self, config, base_class):
        super().__init__()

        self.config     = config
        self.bs         = self.config.getint('BatchSize')
        self.use_augmentation = self.config.getboolean('UseAugmentation')

        print("Loading Data. This might take a while ...")
        self.base_dataset_no_aug = base_class(self.config, augmentation=False)
        if self.use_augmentation:
            self.base_dataset_aug = base_class(self.config, augmentation=True)
            assert len(self.base_dataset_no_aug) == len(self.base_dataset_aug)
        else:
            self.base_dataset_aug = self.base_dataset_no_aug

        indices     = np.random.RandomState(seed=42).permutation(len(self.base_dataset_aug))
        train_end   = int(self.config.getfloat('TrainFrac') * len(self.base_dataset_aug))
        val_end     = train_end + int(self.config.getfloat('ValFrac') * len(self.base_dataset_aug))
        self.train_set  = torch.utils.data.Subset(self.base_dataset_aug, indices[:train_end])
        self.val_set    = torch.utils.data.Subset(self.base_dataset_no_aug, indices[train_end:val_end])
        self.test_set   = torch.utils.data.Subset(self.base_dataset_no_aug, indices[val_end:])


    def setup(self, stage=None):
        print("Setup Stage ", stage)
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.bs, shuffle=True, num_workers=self.config.getint('NumWorkers'))


    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=self.config.getint('NumWorkers'))


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=self.config.getint('NumWorkers'))


    def predict_dataloader(self):
        pass

    #def setup(self, stage=Non)



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('configs.ini')
    config = config['DEFAULT']

    print("Loading Data. This might take a while ...")

    #data_module = UltraDataModule(config=config, base_class=SuperResolutionDataset)
    data_module = UltraDataModule(config=config, base_class=AugmentableUltraDataset)
    #data_module.setup()

    for b, (x, gt) in enumerate(data_module.val_dataloader()):
        for i in range(len(x)):
            plot = np.hstack([x[i].permute(1, 2, 0), gt[i].permute(1, 2, 0)])
            cv2.imshow('pp', plot / 255)
            k = cv2.waitKey()
            if k == 27:
                break
