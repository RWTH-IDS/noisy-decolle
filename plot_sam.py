#!/bin/python
# -----------------------------------------------------------------------------
# File Name : plot_sam.py
# Purpose:
#
# Author: Benedikt Wahl
#
# Creation Date : 11-08-2023
#
# Copyright : (c) Tim Stadtmann, Benedikt Wahl
# Licence : GPLv3
# -----------------------------------------------------------------------------

import sys

sys.path.insert(1, '../') #This lets python search for packages at first in the noisysnns directory

import train_lenet_decolle
from decolle.lenet_decolle_model import LenetDECOLLE, DECOLLELoss, LIFLayerVariableTau, LIFLayer
from decolle.utils import train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib
import quantization.scalar as scalar
#from moviepy.editor import ImageSequenceClip
from PIL import Image as im
from PIL import GifImagePlugin

NUM_RUNS = 3

args, writer, log_dir, checkpoint_dir, test_file = prepare_experiment()


def save_gif (sam_gif, label, layer):
        scale = 1
        sam_gif_npy = sam_gif.cpu().numpy()
        sample_gif = []
        formatted = sam_gif_npy - np.min(sam_gif_npy)
        formatted = (formatted * (255) / np.max(formatted)).astype('uint8')

        for i in range(sam_gif_npy.shape[0]):
            temp_data = im.fromarray(formatted[i,:,:].repeat(40, axis=0).repeat(40, axis=1))
            sample_gif.append(temp_data)

        # Save gif                    
        sample_gif[0].save(args.sam_directory+'/label_'+str(label)+'_layer_'+str(layer)+'.gif', save_all=True, append_images=sample_gif[1:], duration = 10)

if args.sam != -1:
    sam_gif = train_lenet_decolle.main()
    for layer in range (3):
        save_gif(sam_gif[layer], args.sam, layer)
else:
    if args.dataset == 'dvs':
        num_labels = 11
    if args.dataset == 'nmnist':
        num_labels = 10
    for label in range(num_labels):
        args.sam = label
        sam_gif = train_lenet_decolle.main(args)
        for layer in range (3):
            save_gif(sam_gif[layer], label, layer)        




