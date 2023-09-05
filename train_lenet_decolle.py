#!/bin/python
#-----------------------------------------------------------------------------
# File Name : train_lenet_decolle
# Author: Emre Neftci
#
# Creation Date : Sept 2. 2019
# Last Modified : Sept 5. 2023, T. Stadtmann
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------
import sys

sys.path.insert(1, '../') #This lets python search for packages at first in the noisysnns directory

from decolle.lenet_decolle_model import LenetDECOLLE, DECOLLELoss, LIFLayer
from decolle.utils import train, test, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
from decolle.init_functions import init_LSUV
import numpy as np
np.set_printoptions(precision=4)
import torch
import importlib
import quantization.scalar as scalar

def initialize_save_membrane(save_dir):
    temp = []
    temp_torch = torch.Tensor(temp)
    torch.save(temp_torch, save_dir+'_layer_0.pt')
    torch.save(temp_torch, save_dir+'_layer_1.pt')
    torch.save(temp_torch, save_dir+'_layer_2.pt')

def histogramm_save_membrane(save_dir, device):
    membrane_voltage_0 = torch.load(save_dir+'_layer_0.pt', map_location=device)
    membrane_voltage_0 = torch.histogram(membrane_voltage_0.cpu(), 10)
    torch.save((membrane_voltage_0.hist, membrane_voltage_0.bin_edges), save_dir+'_layer_0.pt')

    membrane_voltage_1 = torch.load(save_dir+'_layer_1.pt', map_location=device)
    membrane_voltage_1 = torch.histogram(membrane_voltage_1.cpu(), 10)
    torch.save((membrane_voltage_1.hist, membrane_voltage_1.bin_edges), save_dir+'_layer_1.pt')

    membrane_voltage_2 = torch.load(save_dir+'_layer_2.pt', map_location=device)
    membrane_voltage_2 = torch.histogram(membrane_voltage_2.cpu(), 10)
    torch.save((membrane_voltage_2.hist, membrane_voltage_2.bin_edges), save_dir+'_layer_2.pt')

    
def main():
    args, writer, log_dir, checkpoint_dir, test_file = prepare_experiment()
    
    glob_args = args # Defines global args that can be passed through to other functions without conflict with *args
    starting_epoch = 0

    if args.checkpoint_number >= 1:
        checkpoint_number = args.checkpoint_number - 1 # Substraction needed because first epoch is Epoch1 at position 0 in array
    else:
        checkpoint_number = -1                         # Default value => Last checkpoint

    dataset = importlib.import_module(args.dataset)
    try:
        create_data = dataset.create_data
    except AttributeError:
        create_data = dataset.create_dataloader

    if args.save_voltage != None or args.sam != None:
        args.batch_size = 1

    ## Load Data
    gen_train, gen_test = create_data(chunk_size_train=args.chunk_size_train,
                                    chunk_size_test=args.chunk_size_test,
                                    batch_size=args.batch_size,
                                    dt=args.deltat,
                                    num_workers=args.num_dl_workers)

    data_batch, target_batch = next(iter(gen_train))
    data_batch = torch.Tensor(data_batch).to(args.device)
    target_batch = torch.Tensor(target_batch).to(args.device)

    ## Create Model, Optimizer and Loss
    net = LenetDECOLLE( out_channels=args.out_channels,
                        Nhid=args.Nhid,
                        Mhid=args.Mhid,
                        kernel_size=args.kernel_size,
                        pool_size=args.pool_size,
                        input_shape=args.input_shape,
                        alpha=args.alpha,
                        alpharp=args.alpharp,
                        dropout=args.dropout,
                        beta=args.beta,
                        num_conv_layers=args.num_conv_layers,
                        num_mlp_layers=args.num_mlp_layers,
                        lc_ampl=args.lc_ampl,
                        lif_layer_type = LIFLayer,
                        method=args.learning_method,
                        with_output_layer=args.with_output_layer,
                        threshold=args.threshold).to(args.device)

    if glob_args.quantise_training != 0:
        scalar.quantize_model_(net, bits=glob_args.quantise_training, percentile=glob_args.percentile, p=glob_args.p_quantise, method=glob_args.quant_method)

    if hasattr(args.learning_rate, '__len__'):
        from decolle.utils import MultiOpt
        opts = []
        for i in range(len(args.learning_rate)):
            opts.append(torch.optim.Adamax(net.get_trainable_parameters(i), lr=args.learning_rate[i], betas=args.betas, weight_decay=glob_args.L2))
        opt = MultiOpt(*opts)
    else:
        opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=glob_args.L2)

    if args.loss_scope=='global':
        loss = [None for i in range(len(net))]
        if net.with_output_layer: 
            loss[-1] = cross_entropy_one_hot
        else:
            raise RuntimeError('bptt mode needs output layer')
        decolle_loss = DECOLLELoss(net = net, loss_fn = loss, smooth_fct = args.reg_smooth_fct, reg1_l=args.reg1_l, reg2_l=args.reg2_l)
    else:
        loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]
        if net.with_output_layer:
            loss[-1] = cross_entropy_one_hot
        decolle_loss = DECOLLELoss(net = net, loss_fn = loss, smooth_fct = args.reg_smooth_fct, reg1_l=args.reg1_l, reg2_l=args.reg2_l)

    ## Initialize
    net.init_parameters(data_batch[:32])            # Initialises weights and biases without running decolle
    init_LSUV(net, data_batch[:32])                 # Initialises weights and biases

    ## Resume if necessary
    if args.resume_from is not None:
        starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt, checkpoint_number)

    # --------TRAINING LOOP----------
    if not args.no_train:
        print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))
        if args.resume_from is not None:
            print('Resuming from epoch {} with lr {}'.format(starting_epoch, opt.param_groups[-1]['lr']))

        test_acc_hist = []
        for e in range(starting_epoch , args.num_epochs):
            interval = e // args.lr_drop_interval
            
            glob_args.mismatch_forward = args.mismatch_forward / (np.power(args.lr_drop_factor,e/args.lr_drop_interval))

            if interval > 0:
                opt.param_groups[-1]['lr'] = np.array(args.learning_rate) / (interval * args.lr_drop_factor)
                glob_args.mismatch =  args.mismatch / (interval * args.lr_drop_factor)

                print('Changing learning rate to {}'.format(opt.param_groups[-1]['lr']))
            else:
                opt.param_groups[-1]['lr'] = np.array(args.learning_rate)

            if (e % args.test_interval) == 0 and e!=0:
                print('---------------Epoch {}-------------'.format(e))
                if not args.no_save:
                    print('---------Saving checkpoint---------')
                    save_checkpoint(e, checkpoint_dir, net, opt)

                test_loss, test_acc = test(gen_test, decolle_loss, net, args.burnin_steps, print_error = True)
                test_acc_hist.append(test_acc)

                if not args.no_save:
                    write_stats(e, test_acc, test_loss, writer)
                    np.save(log_dir+'/test_acc.npy', np.array(test_acc_hist),)

            total_loss, act_rate = train(gen_train, decolle_loss, net, opt, e, args.burnin_steps, online_update=args.online_update, glob_args=glob_args)
            if not args.no_save:
                for i in range(len(net)):
                    writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)
    else:
        if glob_args.membrane_voltage_save_dir != None:
            initialize_save_membrane(glob_args.membrane_voltage_save_dir)

        if glob_args.quantise_test != 0:
            scalar.quantize_model_(net, bits=glob_args.quantise_test, percentile=glob_args.percentile, p=1.0, method=glob_args.quant_method)

        if glob_args.save_voltage != None:
            return test(gen_test, decolle_loss, net, args.burnin_steps, glob_args, print_error = True)

        if glob_args.sam != None:
            return test(gen_test, decolle_loss, net, args.burnin_steps, glob_args, print_error = True)

        test_acc_hist = []
        test_loss, test_acc = test(gen_test, decolle_loss, net, args.burnin_steps, glob_args, print_error = True)
        test_acc_hist.append(test_acc)

        np.save(test_file, np.array(test_acc_hist))

if __name__ == "__main__":
    main()