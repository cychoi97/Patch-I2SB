import os
import glob
import cv2
import numpy as np
import pydicom

from PIL import Image
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import natsort
from torch.utils.data import Dataset
from torchvision import transforms as T


EXTENSION = ['.jpg', '.jpeg', '.png', '.tiff', '.dcm']


class Dataset(Dataset):
    def __init__(self, opt, log, mode):
        super().__init__()
        self.mode = mode
        self.dataset_dir = opt.dataset_dir / self.mode
        self.src = opt.src
        self.trg = opt.trg
        self.image_size = opt.image_size

        # Input
        self.src_fnames = {os.path.relpath(os.path.join(root, fname), start=self.dataset_dir / opt.src) for root, _dirs, files in os.walk(self.dataset_dir / opt.src) for fname in files}
        self.src_image_fnames = natsort.natsorted(fname for fname in self.src_fnames if self._file_ext(fname) in EXTENSION)
        if len(self.src_image_fnames) == 0:
            raise IOError('No source image files found in the specified path')
        log.info(f"[Dataset] Built dataset {self.dataset_dir}, size={len(self.src_image_fnames)}!")
        
        # Output
        self.trg_fnames = {os.path.relpath(os.path.join(root, fname), start=self.dataset_dir / opt.trg) for root, _dirs, files in os.walk(self.dataset_dir / opt.trg) for fname in files}
        self.trg_image_fnames = natsort.natsorted(fname for fname in self.trg_fnames if self._file_ext(fname) in EXTENSION)
        if len(self.trg_image_fnames) == 0:
            raise IOError('No target image files found in the specified path')
        log.info(f"[Dataset] Built dataset {self.dataset_dir}, size={len(self.trg_image_fnames)}!")
            
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1,1]
        ])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def _open_file(self, domain, fname):
        return open(os.path.join(self.dataset_dir, domain, fname), 'rb')
    
    def _padding(self, img):
        if img.shape[0] != img.shape[1]:
            if img.shape[0] > img.shape[1]:
                padding = np.zeros((img.shape[0], (img.shape[0] - img.shape[1]) // 2), np.float32)
                img = np.concatenate([padding, img, padding], 1)
            elif img.shape[0] < img.shape[1]:
                padding = np.zeros(((img.shape[1] - img.shape[0]) // 2, img.shape[1]), np.float32)
                img = np.concatenate([padding, img, padding], 0)
        else:
            pass
        return img

    def _resize(self, img):
        if img.shape[0] < self.image_size or img.shape[1] < self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), cv2.INTER_CUBIC)
        elif img.shape[0] > self.image_size or img.shape[1] > self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), cv2.INTER_AREA)
        return img

    def _clip_and_normalize(self, img, min, max):
        img = np.clip(img, min, max)
        img = (img - min) / (max - min)
        return img
    
    def _CT_preprocess(self, dcm, img, window_width=None, window_level=None):
        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        img = img * slope + intercept

        if window_width is not None and window_level is not None:
            min = window_level - (window_width / 2.0)
            max = window_level + (window_width / 2.0)
        else: # 12 bits
            min = -1024.0
            max = 3071.0

        # img = self._padding(img) # zero padding for preserving image ratio
        # img = self._resize(img)
        img = self._clip_and_normalize(img, min, max)
        return img

    def __len__(self):
        return len(self.src_image_fnames)

    def __getitem__(self, index):
        src_fname = self.src_image_fnames[index]
        trg_fname = self.trg_image_fnames[index]

        with self._open_file(self.src, src_fname) as f:
            if self._file_ext(src_fname) == '.dcm':
                src_dcm = pydicom.dcmread(f, force=True)
                src_img = src_dcm.pixel_array.astype(np.float32)
                src_img = self._CT_preprocess(src_dcm, src_img, None, None)
            else: # jpg, jpeg, tiff, png, etc.
                src_img = np.array(Image.open(f)).astype(np.float32)

        with self._open_file(self.trg, trg_fname) as f:
            if self._file_ext(trg_fname) == '.dcm':
                trg_dcm = pydicom.dcmread(f, force=True)
                trg_img = trg_dcm.pixel_array.astype(np.float32)
                trg_img = self._CT_preprocess(trg_dcm, trg_img, None, None)
            else: # jpg, jpeg, tiff, png, etc.
                trg_img = np.array(Image.open(f)).astype(np.float32)

        corrupt_img = self.transform(src_img[:,:,np.newaxis])
        clean_img = self.transform(trg_img[:,:,np.newaxis])

        return clean_img, corrupt_img