import os
import os.path
import torch
import numpy as np
import pandas
from glob import glob
import random
from collections import OrderedDict
import cv2
import numpy as np

def X2Cube(img, cellSize=4):

    B = [cellSize, cellSize]
    skip = [cellSize, cellSize]
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//cellSize, N//cellSize, cellSize*cellSize)

    hsi_min = np.min(DataCube, axis=(0,1), keepdims=True)
    hsi_max = np.max(DataCube, axis=(0,1), keepdims=True)
    hsi_normed = (DataCube - hsi_min) / ((hsi_max - hsi_min) + 1e-6) * 255
    hsi_normed = hsi_normed.astype(np.uint8)
    return hsi_normed


def X2Cube_fast(img, cellSize=4):
    M, N = img.shape
    if (M % cellSize) != 0 or (N % cellSize) != 0:
        raise ValueError(f"Image shape {(M, N)} must be divisible by cellSize {cellSize}")

    cubes = (
        img.reshape(M // cellSize, cellSize, N // cellSize, cellSize)
        .swapaxes(1, 2)
        .reshape(M // cellSize, N // cellSize, cellSize * cellSize)
    )

    hsi_min = np.min(cubes, axis=(0, 1), keepdims=True)
    hsi_max = np.max(cubes, axis=(0, 1), keepdims=True)
    hsi_normed = (cubes - hsi_min) / ((hsi_max - hsi_min) + 1e-6) * 255
    return hsi_normed.astype(np.uint8)


def get_image_loader(img_file: str, cellSize=-1) -> np.array:
    if cellSize > 0:
        img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = X2Cube(img, cellSize)
    elif cellSize == -1:
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    else:
        raise Exception
    return img


def _get_frames(self, seq_id, frame_ids):
    frame_list = []
    for frame_id in frame_ids:
        frame_name = '%04d.png' % (frame_id+1)
        image_path = os.path.join(self.RGBHDataRootDir, self.sequence_list[seq_id], 'img', frame_name)
        if self.cur_modalName == 'NIR':
            hsi_image = self.get_image_loader(image_path, cellSize=5)
        elif self.cur_modalName == 'VIS':
            hsi_image = self.get_image_loader(image_path, cellSize=4)
        elif self.cur_modalName == 'RedNIR':
            hsi_image = self.get_image_loader(image_path, cellSize=4)
            hsi_image = hsi_image[:,:,:-1]

        frame_name = '%04d.jpg' % (frame_id+1)

        image_path = os.path.join(self.RGBHDataRootDir.replace('HSI-'+self.cur_modalName, 'HSI-'+self.cur_modalName+'-FalseColor'), self.sequence_list[seq_id], 'img', frame_name)
        hsi_fc_image = self.get_image_loader(image_path, cellSize=-1)
        frame_list.append(np.concatenate((hsi_fc_image, hsi_image), axis=2))

    return frame_list 


def get_frames(self, seq_id, frame_ids, anno=None):
    frame_list = self._get_frames(seq_id, frame_ids)

    if anno is None:
        anno = self.get_sequence_info(seq_id)

    anno_frames = {}
    for key, value in anno.items():
        if key == 'seq_belong_mask':
            continue
        anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

    object_meta = OrderedDict({'object_class_name': None,
                                'motion_class': None,
                                'major_class': None,
                                'root_class': None,
                                'motion_adverb': None})

    return frame_list, anno_frames, object_meta

