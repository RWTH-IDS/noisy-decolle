#!/bin/python
# -----------------------------------------------------------------------------
# File Name : voltage_histograms.py
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
from decolle.utils import parse_args
import numpy as np

NUM_RUNS = 3

args = parse_args('parameters/params.yml') 
temp_resume_from = args.resume_from

args.save_voltage = 0
u_layer0 = np.array([])
for i in range(1,NUM_RUNS+1):
    args.resume_from = temp_resume_from.replace("NUMBER", str(i))
    u_layer0 = np.append(u_layer0, train_lenet_decolle.main(args))

if args.dataset == 'dvs':
    hist_layer0 = np.histogram(u_layer0, bins=[-10+0.5*x for x in range(40+1) ])
    sample_size_layer0 = len(u_layer0)
    hist_layer0 = [hist_layer0[0].astype('float32') / float(sample_size_layer0), hist_layer0[1]]
    std_dev_layer0 = np.nanstd(np.clip(u_layer0, -10e300, 10e300))
    proportion_spiking_layer0 = len(np.nonzero([u_layer0 > 0][0])[0]) / float(sample_size_layer0)
    proportion_displayed_layer0 = len(np.nonzero([u_layer0 > -10][0] & [u_layer0 < 10][0])[0]) / float(sample_size_layer0)
    layer0 = [hist_layer0[0], hist_layer0[1], sample_size_layer0, std_dev_layer0, proportion_spiking_layer0, proportion_displayed_layer0]
    
elif args.dataset == 'nmnist':
    hist_layer0 = np.histogram(u_layer0, bins=[-40+2*x for x in range(40+1)])
    sample_size_layer0 = len(u_layer0)
    hist_layer0 = [hist_layer0[0].astype('float32') / float(sample_size_layer0), hist_layer0[1]]
    std_dev_layer0 = np.nanstd(np.clip(u_layer0, -10e300, 10e300))
    proportion_spiking_layer0 = len(np.nonzero([u_layer0 > 0][0])[0]) / float(sample_size_layer0)
    proportion_displayed_layer0 = len(np.nonzero([u_layer0 > -40][0] & [u_layer0 < 40][0])[0]) / float(sample_size_layer0)
    layer0 = [hist_layer0[0], hist_layer0[1], sample_size_layer0, std_dev_layer0, proportion_spiking_layer0, proportion_displayed_layer0]

np.save(args.voltage_save_dir+'_layer_0.npy', np.array(layer0, dtype=object), allow_pickle=True)
del u_layer0

args.save_voltage = 1
u_layer1 = np.array([])
for i in range(1,NUM_RUNS+1):
    args.resume_from = temp_resume_from.replace("NUMBER", str(i))
    u_layer1 = np.append(u_layer1, train_lenet_decolle.main(args))
if args.dataset == 'dvs':
    hist_layer1 = np.histogram(u_layer1, bins=[-10+0.5*x for x in range(40+1) ])
    sample_size_layer1 = len(u_layer1)
    hist_layer1 = [hist_layer1[0].astype('float32') / float(sample_size_layer1), hist_layer1[1]]
    std_dev_layer1 = np.nanstd(np.clip(u_layer1, -10e300, 10e300))
    proportion_spiking_layer1 = len(np.nonzero([u_layer1 > 0][0])[0]) / float(sample_size_layer1)
    proportion_displayed_layer1 = len(np.nonzero([u_layer1 > -10][0] & [u_layer1 < 10][0])[0]) / float(sample_size_layer1)
    layer1 = [hist_layer1[0], hist_layer1[1], sample_size_layer1, std_dev_layer1, proportion_spiking_layer1, proportion_displayed_layer1]
    
elif args.dataset == 'nmnist':
    hist_layer1 = np.histogram(u_layer1, bins=[-150+7.5*x for x in range(40+1)])
    sample_size_layer1 = len(u_layer1)
    hist_layer1 = [hist_layer1[0].astype('float32') / float(sample_size_layer1), hist_layer1[1]]
    std_dev_layer1 = np.nanstd(np.clip(u_layer1, -10e300, 10e300))
    proportion_spiking_layer1 = len(np.nonzero([u_layer1 > 0][0])[0]) / float(sample_size_layer1)
    proportion_displayed_layer1 = len(np.nonzero([u_layer1 > -150][0] & [u_layer1 < 150][0])[0]) / float(sample_size_layer1)
    layer1 = [hist_layer1[0], hist_layer1[1], sample_size_layer1, std_dev_layer1, proportion_spiking_layer1, proportion_displayed_layer1]

np.save(args.voltage_save_dir+'_layer_1.npy', np.array(layer1, dtype=object), allow_pickle=True)
del u_layer1


u_layer2 = np.array([])
for i in range(1,NUM_RUNS+1):
    args.resume_from = temp_resume_from.replace("NUMBER", str(i))
    u_layer2 = np.append(u_layer2, train_lenet_decolle.main(args))
u_layer2 = train_lenet_decolle.main(args)
if args.dataset == 'dvs':
    hist_layer2 = np.histogram(u_layer2, bins=[-10+0.5*x for x in range(40+1) ])
    sample_size_layer2 = len(u_layer2)
    hist_layer2 = [hist_layer2[0].astype('float32') / float(sample_size_layer2), hist_layer2[1]]
    std_dev_layer2 = np.nanstd(np.clip(u_layer2, -10e300, 10e300))
    proportion_spiking_layer2 = len(np.nonzero([u_layer2 > 0][0])[0]) / float(sample_size_layer2)
    proportion_displayed_layer2 = len(np.nonzero([u_layer2 > -10][0] & [u_layer2 < 10][0])[0]) / float(sample_size_layer2)
    layer2 = [hist_layer2[0], hist_layer2[1], sample_size_layer2, std_dev_layer2, proportion_spiking_layer2, proportion_displayed_layer2]
    
elif args.dataset == 'nmnist':
    hist_layer2 = np.histogram(u_layer2, bins=[-300+15*x for x in range(40+1)])
    sample_size_layer2 = len(u_layer2)
    hist_layer2 = [hist_layer2[0].astype('float32') / float(sample_size_layer2), hist_layer2[1]]
    std_dev_layer2 = np.nanstd(np.clip(u_layer2, -10e300, 10e300))
    proportion_spiking_layer2 = len(np.nonzero([u_layer2 > 0][0])[0]) / float(sample_size_layer2)
    proportion_displayed_layer2 = len(np.nonzero([u_layer2 > -300][0] & [u_layer2 < 300][0])[0]) / float(sample_size_layer2)
    layer2 = [hist_layer2[0], hist_layer2[1], sample_size_layer2, std_dev_layer2, proportion_spiking_layer2, proportion_displayed_layer2]

np.save(args.voltage_save_dir+'_layer_2.npy', np.array(layer2, dtype=object), allow_pickle=True)
del u_layer2

print("Done")
