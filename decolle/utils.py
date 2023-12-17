import argparse 
import torch 
from torchvision import datasets, models, transforms, utils
from torch.utils.data.dataloader import DataLoader
import numpy as np 
import tqdm 
import datetime 
import os 
import socket 
import argparse 
import math 
from collections import Counter, defaultdict
from tensorboardX import SummaryWriter 
import random
import warnings
try:
    import matplotlib
    import matplotlib.pyplot as plt
except:
    print("WARN: Couldn't load matplotlib")
    pass

from PIL import Image as im

relu = torch.relu 
sigmoid = torch.sigmoid 

def state_detach(state):
    for s in state:
        s.detach_()

def cross_entropy_one_hot(input, target): 
    _, labels = target.max(dim=1) 
    return torch.nn.CrossEntropyLoss()(input, labels) 

def spiketrains(N, T, rates, mode = 'poisson'):
    '''
    *N*: number of neurons
    *T*: number of time steps
    *rates*: vector or firing rates, one per neuron
    *mode*: 'regular' or 'poisson'
    '''
    def __gen_ST(N, T, rate, mode = 'regular'):    
        if mode == 'regular':
            spikes = np.zeros([T, N])
            spikes[::(1000//rate)] = 1
            return spikes
        elif mode == 'poisson':
            spikes = np.ones([T, N])        
            spikes[np.random.binomial(1,float(1000. - rate)/1000, size=(T,N)).astype('bool')] = 0
            return spikes
        else:
            raise Exception('mode must be regular or Poisson')

    if not hasattr(rates, '__iter__'):
        return __gen_ST(N, T, rates, mode)
    rates = np.array(rates)
    M = rates.shape[0]
    spikes = np.zeros([T, N])
    for i in range(M):
        if int(rates[i])>0:
            spikes[:,i] = __gen_ST(1, T, int(rates[i]), mode = mode).flatten()
    return spikes

def plotLIF(U, S, Vplot = 'all', staggering= 1, ax1=None, ax2=None, **kwargs):
    '''
    This function plots the output of the function LIF.
    
    Inputs:
    *S*: an TxNnp.array, where T are time steps and N are the number of neurons
    *S*: an TxNnp.array of zeros and ones indicating spikes. This is the second
    output return by function LIF
    *Vplot*: A list indicating which neurons' membrane potentials should be 
    plotted. If scalar, the list range(Vplot) are plotted. Default: 'all'
    *staggering*: the amount by which each V trace should be shifted. None
    
    Outputs the figure returned by figure().    
    '''
    def spikes_to_evlist(spikes):
        t = np.tile(np.arange(spikes.shape[0]), [spikes.shape[1],1])
        n = np.tile(np.arange(spikes.shape[1]), [spikes.shape[0],1]).T  
        return t[spikes.astype('bool').T], n[spikes.astype('bool').T]

    V = U
    spikes = S
    #Plot
    t, n = spikes_to_evlist(spikes)
    #f = plt.figure()
    if V is not None and ax1 is None:
        ax1 = plt.subplot(211)
    elif V is None:
        ax1 = plt.axes()
        ax2 = None
    ax1.plot(t, n, 'k|', **kwargs)
    ax1.set_ylim([-1, spikes.shape[1] + 1])
    ax1.set_xlim([0, spikes.shape[0]])

    if V is not None:
        if Vplot == 'all':
            Vplot = range(V.shape[1])
        elif not hasattr(Vplot, '__iter__'):
            Vplot = range(np.minimum(Vplot, V.shape[1]))    
        
        if ax2 is None:
            ax2 = plt.subplot(212)
    
        if V.shape[1]>1:
            for i, idx in enumerate(Vplot):
                ax2.plot(V[:,idx]+i*staggering,'-',  **kwargs)
        else:
            ax2.plot(V[:,0], '-', **kwargs)
            
        if staggering!=0:
            plt.yticks([])
        plt.xlabel('time [ms]')
        plt.ylabel('u [au]')

    ax1.set_ylabel('Neuron ')

    plt.xlim([0, spikes.shape[0]])
    plt.ion()
    plt.show()
    return ax1,ax2

def image2spiketrain(x,y,gain=50,min_duration=None, max_duration=500):
    def to_one_hot(t, width):
        t_onehot = torch.zeros(*t.shape+(width,))
        return t_onehot.scatter_(1, t.unsqueeze(-1), 1)

    y = to_one_hot(y, 10)
    if min_duration is None:
        min_duration = max_duration-1
    batch_size = x.shape[0]
    T = np.random.randint(min_duration,max_duration,batch_size)
    Nin = np.prod([28,28,1])
    allinputs = np.zeros([batch_size,max_duration, Nin])
    for i in range(batch_size):
        st = spiketrains(T = T[i], N = Nin, rates=gain*x[i].reshape(-1)).astype(np.float32)
        allinputs[i] =  np.pad(st,((0,max_duration-T[i]),(0,0)),'constant')
    allinputs = np.transpose(allinputs, (1,0,2))
    allinputs = allinputs.reshape(allinputs.shape[0],allinputs.shape[1],1, 28,28)

    alltgt = np.zeros([max_duration, batch_size, 10], dtype=np.float32)
    for i in range(batch_size):
        alltgt[:,i,:] = y[i]

    return allinputs, alltgt

def plot_sample_spatial(data, target, figname): 
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    matplotlib.use('Agg') 
    image = np.sum(tonp(data), axis=(0,1)) # sum over time and polarity 
    fig, ax = plt.subplots() 
    im = ax.imshow(image) 
    divider = make_axes_locatable(ax) 
    cax = divider.append_axes("right", size="5%", pad=0.05) 
    cbar = plt.colorbar(im, cax=cax) 
    ax.set_title('Sample for label {}'.format(tonp(target[0]))) 
    plt.savefig(figname) 
 
def plot_sample_temporal(data, target, figname): 
    def smooth(y, box_pts): 
        box = np.ones(box_pts)/box_pts 
        y_smooth = np.convolve(y, box, mode='same') 
        return y_smooth 
    
    events = np.sum(tonp(data), axis=(1,2,3)) # sum over polarity, x and y 
    fig, ax = plt.subplots() 
    ax.set_title('Sample for label {}'.format(tonp(target[0]))) 
    ax.plot(smooth(events, 20)) # smooth the signal with temporal convolution 
    plt.savefig(figname) 
 
def print_to_gif(data_batch, target_batch, dtype, device, glob_args, batch_num, noise, epoch_num=None):
    home = os.path.expanduser("~")
    path = os.path.join(home+glob_args.gif_save_dir, noise)
    
    # Creating a directory to store the gifs within the given parent directory
    if not os.path.exists(path):
        os.mkdir(path)

    path_npy = os.path.join(path, "num_gifs.npy")

    # Create a npy file within the new subfolder to keep track and limit the number of gifs per category and type of noise
    if not os.path.exists(path_npy):
        num_gifs = np.zeros(target_batch.shape[2], dtype=int)
    else:
        num_gifs = np.load(path_npy)

    # Acertain the tensor type is set to device
    data_batch = torch.tensor(data_batch).type(dtype).to(device) 
    target_batch = torch.tensor(target_batch).type(dtype).to(device) 
    
    # Splitting the chanels into two tensors and convert them to numpy arrays
    off_tensor = data_batch[:,:,0,:,:]
    on_tensor = data_batch[:,:,1,:,:]
    off_npy = off_tensor.clone().cpu().detach().numpy()
    on_npy = on_tensor.clone().cpu().detach().numpy()

    # Create a difference array of on and off channel with all values >= 0            
    ones = np.ones((off_tensor.shape[0], off_tensor.shape[1],off_tensor.shape[2],off_tensor.shape[3]))
    diff_npy = ones + on_npy - off_npy

    # Formatting the difference vector for image representation
    formatted = (diff_npy * 255 / 2).astype('uint8')
                
    for sample in range(off_tensor.shape[0]):
        sample_gif = []
        
        # Evaluate category of sample
        category = np.argmax(target_batch[sample,0,:].clone().cpu().detach().numpy())

        # limit the number of gifs per category and type of noise
        if num_gifs[category] >= 5:
            continue
        
        # Create image in every timestep from upscaled array and append it to gif
        for i in range(off_tensor.shape[0]):
            temp_data = im.fromarray(formatted[sample,i,:,:].repeat(10, axis=0).repeat(10, axis=1))
            sample_gif.append(temp_data)

        # Save gif                    
        sample_gif[0].save(path+"/batch_"+str(batch_num)+"_sample_"+str(sample)+"_category_"+str(category)+".gif", save_all=True, append_images=sample_gif[1:], duration = 10)

        # keep track the number of gifs per category and type of noise
        num_gifs[category] += 1

    np.save(path_npy, num_gifs)

def tonp(tensor): 
    if type(tensor) == type([]): 
        return [t.detach().cpu().numpy() for t in tensor] 
    elif not hasattr(tensor, 'detach'): 
        return tensor 
    else: 
        return tensor.detach().cpu().numpy() 
 
def prepare_experiment():
    # extract parameters
    parser = argparse.ArgumentParser(description='DECOLLE for event-driven object recognition') 
    parser.add_argument('--device', type=str, help='Device to use (cpu or cuda)') 
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--resume_from', type=str, metavar='path_to_logdir', help='Path to a previously saved checkpoint') 
    parser.add_argument('--params_file', type=str, help='Path to parameters file to load. Ignored if resuming from checkpoint') 
    parser.add_argument('--no_save', type=bool, help='Set this flag if you don\'t want to save results') 
    parser.add_argument('--save_dir', type=str, default='', help='Name of subdirectory to save results in') 
    parser.add_argument('--verbose', type=bool, help='print verbose outputs') 
    parser.add_argument('--seed', type=int, help='CPU and GPU seed') 
    parser.add_argument('--no_train', type=bool, help='Solely test model (requires resume_from and checkpoint_number)')
    parser.add_argument('--checkpoint_number', type=int, help='Checkpoint to start at')
    parser.add_argument('--thermal_noise', type=float, help='Standard deviation of Gaussian noise on membrane simulating thermal noise')
    parser.add_argument('--hot_pixels', type=float, help='Hot pixels proportion (in %)')
    parser.add_argument('--ba_noise', type=float, help='Lambda of background activity')
    parser.add_argument('--ba_noise_torch', type=float, help='Lambda of background activity (faster computation, slightly different)')
    parser.add_argument('--gif_save_dir', type=str, help='Enter gif parent save dir starting with / if test data should be saved as gif')
    parser.add_argument('--membrane_voltage_save_dir', type=str, help='Save membrane voltages to this directory (add name without .npy)')
    parser.add_argument('--mismatch', type=float, help='Enter level of mismatch in percent')
    parser.add_argument('--spike_loss', type=float, help='Define percentage of spike loss from one layer to another')
    parser.add_argument('--spike_add', type=float, help='Define percentage of added spikes from one layer to another')
    parser.add_argument('--weight_bias_save_dir', type=str, help= 'Save weight and bias distribution to this directory (add name without .npy) coming from home directory')
    parser.add_argument('--percentile', type=float, help='Percentile for calculating s for quantisation')   
    parser.add_argument('--log_folder', type=str, help='Enter name for log folder')
    parser.add_argument('--dropout', type=float, help='Enter the dropout value ranging from 0 to 1')
    parser.add_argument('--mismatch_forward', type=float, help='Forward weight noise during training')
    parser.add_argument('--quantise_bits', type=int, help='Enter number of bits to quantise') 
    parser.add_argument('--p_quantise', type=float, help="Enter the amount of weights quantised every step")
    parser.add_argument('--initial_lr_drop', type=int, help='Initial lr drop')
    parser.add_argument('--quant_method', type=str, help= 'Enter quantisation method for training')
    parser.add_argument('--L2', type=float, help='Enter L2 regularisatio parameter')
    parser.add_argument('--ba_train', type=float, help='BA value for training')
    parser.add_argument('--symmetric', type=bool, help='Set if fairseq quantisation should be symmetric')
    parser.add_argument('--save_voltage', type=int, help='Enter layer to be saved from 0 to 2')
    parser.add_argument('--voltage_save_dir', type=str, help='Enter voltage save dir')
    parser.add_argument('--dataset', type=str, help='enter dataset: dvs or nmnist')
    parser.add_argument('--sam', type=int, help='Enter label of which you want to have SAM')
    parser.add_argument('--sam_directory', type=str, help='Enter directory for sam plots')
    parser.add_argument('--reg1_l', nargs='*', type=float, help='l1 reg factors -> keeps firing low (per layer)')
    parser.add_argument('--reg2_l', nargs='*', type=float, help='l2 reg factors -> enables some firing (per layer)')
    parser.add_argument('--threshold', type=float, help='spiking threshold')
    parsed, unknown = parser.parse_known_args() 
 
    for arg in unknown: 
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=str)
 
    args = parser.parse_args() 
      
    if args.no_save: 
        print('!!!!WARNING!!!!\n\nRESULTS OF THIS TRAINING WILL NOT BE SAVED\n\n!!!!WARNING!!!!\n\n') 

    # setup default directories for logging and saving, load  yaml file, setup tensorboard, set seeds
    test_file = ''
    test_params_file = ''
    if args.resume_from is None: 
        params_file = args.params_file 
        if not args.no_save: 
            current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S') 
            if args.log_folder != None:
                log_dir = os.path.join('logs/', args.save_dir, args.log_folder) 
            else:
                log_dir = os.path.join('logs/', args.save_dir + '_' + current_time + '_' + socket.gethostname()) 
            checkpoint_dir = os.path.join(log_dir, 'checkpoints') 
            if not os.path.exists(log_dir): 
                os.makedirs(log_dir) 
            writer = SummaryWriter(log_dir=log_dir) 
            print('Results directory: {}'.format(log_dir)) 
    else: 
        log_dir = args.resume_from 
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        if not args.no_save: 
            writer = SummaryWriter(log_dir=log_dir) 
        params_file = os.path.join(log_dir, 'params.yml')
        print('Resuming model from {}'.format(log_dir)) 
        if args.no_train:
            if args.save_dir == '':
                args.save_dir = 'test_acc'
            test_file = os.path.join(log_dir, args.save_dir+".npy")
            test_params_file = os.path.join(log_dir, 'params_'+args.save_dir+'.yml')

    with open(params_file, 'r') as f: 
        import yaml 
        params = yaml.safe_load(f)
        if 'save_dir' in params and params['save_dir']!='' and params['save_dir']!=None and args.resume_from is not None:
            print("Warning! Found save_dir entry in " + params_file + " - will be ignored, only writable via user input!")

    for (k,v) in vars(args).items():  # set all values that are defined in args but not in params (parameter file) to default values ("" for str, 0 otherwise)
        if k not in params:
            default_val = 0 if type(getattr(args, k)) != str else ""
            warnings.warn("Missing parameter in parameters file: " + k + ". Setting to " + str(default_val) + ".")
            setattr(args, k, default_val)
    args = {k:v for (k,v) in vars(args).items() if v is not None} # filter out all 'None' entries from args (args now only contains explicit user inputs)
    args = {**params, **args} # merge args and params dictionaries into single dictionaries for all values (entries in args override entries in params)
    args_=argparse.Namespace()
    for k,v in args.items():
        setattr(args_, k, v)
    args=args_ # save args again as namespace object, so we can do args.value instead of args['value']

    params_dump = os.path.join(log_dir, 'params.yml') if test_params_file=='' else test_params_file
    with open(params_dump, 'w') as f:
        yaml.dump(vars(args), f)
        print("Params directory: {}".format(params_dump))

    if args.no_save:
        write=None
        if args.resume_from is None:
            log_dir=''
            checkpoint_dir=''

    if log_dir!='':
        print("Checkpoint directory: " + checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
 
    if args.seed != -1: 
        print("Setting seed {0}".format(args.seed)) 
        torch.manual_seed(args.seed) 
        np.random.seed(args.seed) 

    # Printing parameters
    if args.verbose:
        print('Using the following parameters:')
        for k,v in vars(args).items():
            print(k, ": ", v)
    
    #args = defaultdict(lambda: None,args)
    return args, writer, log_dir, checkpoint_dir, test_file
 
def load_model_from_checkpoint(checkpoint_dir, net, opt, n_checkpoint=-1, device='cuda'): 
    ''' 
    checkpoint_dir: string containing path to checkpoints, as stored by save_checkpoint 
    net: torch module with state_dict function 
    opt: torch optimizers 
    n_checkpoint: which checkpoint to use. number is not epoch but the order in the ordered list of checkpoint files 
    device: device to use (TODO: get it automatically from net) 
    ''' 
    starting_epoch = 0 
    checkpoint_list = os.listdir(checkpoint_dir) 
    if checkpoint_list: 
        checkpoint_list.sort() 
        last_checkpoint = checkpoint_list[n_checkpoint] 
        checkpoint = torch.load(os.path.join(checkpoint_dir, last_checkpoint), map_location=device) 
        net.load_state_dict(checkpoint['model_state_dict']) 
        opt.load_state_dict(checkpoint['optimizer_state_dict']) 
        starting_epoch = checkpoint['epoch'] 
    return starting_epoch 
 
def train(gen_train, decolle_loss, net, opt, epoch, burnin, online_update=True, glob_args=None, batches_per_epoch=-1): 
    ''' 
    Trains a DECOLLE network 
 
    Arguments: 
    gen_train: a dataloader 
    decolle_loss: a DECOLLE loss function, as defined in base_model 
    net: DECOLLE network 
    opt: optimizaer 
    epoch: epoch number, for printing purposes only 
    burnin: time during which the dynamics will be run, but no updates are made 
    online_update: whether updates should be made at every timestep or at the end of the sequence. 
    ''' 
    device = net.get_input_layer_device() 
    iter_gen_train = iter(gen_train) 
    total_loss = np.zeros(decolle_loss.num_losses) 
    act_rate = [0 for i in range(len(net))] 
 
         
    loss_tv = torch.tensor(0.).to(device) 
    net.train() 
    if hasattr(net.LIF_layers[0], 'base_layer'): 
        dtype = net.LIF_layers[0].base_layer.weight.dtype 
    else: 
        dtype = net.LIF_layers[0].weight.dtype 
    batch_iter = 0 
     
    for data_batch, target_batch in tqdm.tqdm(iter_gen_train, desc='Epoch {}'.format(epoch)): 
        data_batch = torch.tensor(data_batch).type(dtype).to(device) 
        target_batch = torch.tensor(target_batch).type(dtype).to(device) 
        if len(target_batch.shape) == 2: 
            target_batch = target_batch.unsqueeze(1) 
            shape_with_time = np.array(target_batch.shape) 
            shape_with_time[1] = data_batch.shape[1] 
            target_batch = target_batch.expand(*shape_with_time) 
 
        loss_mask = (target_batch.sum(2)>0).unsqueeze(2).float()
        
        # Noise: Background Activity
        if glob_args!=None and glob_args.ba_train > 0.0:
            # For runtime reasons the number of affected pixels is only computed once per batch instead of every timestep.
            # Should be discrete equivalent to Gaussian input noise es regularization method
            # Due to discrete character, we add and delete ones
            # Additional Feature #1: for every batch the number of affected pixels is drawn from a poisson distribution
            # Additional Feature #2: By dividing through the number of ones / zeroes it can be guaranteed that there is an equal number of ones added / deleted
            num_ba_pixels = np.random.poisson(glob_args.ba_train) # Calculate number of affected pixel
            torch.cuda.empty_cache()
            # Calculate the chance that one pixel of a sample at a timestamp is changed from 0 to 1 (expected value per sample: num_ba_pixels)
            chance_add_per_channel = (num_ba_pixels*data_batch.shape[0]*data_batch.shape[1]) / (torch.nonzero(torch.logical_xor(torch.ones(data_batch.shape).to(data_batch.dtype).to(data_batch.device), data_batch).to(data_batch.dtype).to(data_batch.device)).shape[0])
            torch.cuda.empty_cache()
            # Calculate the chance that one pixel of a sample at a timestamp is changed from 1 to 0 (expected value per sample: num_ba_pixels)
            chance_delete_per_channel = (num_ba_pixels*data_batch.shape[0]*data_batch.shape[1]) / (torch.nonzero(data_batch).shape[0])
            torch.cuda.empty_cache()
            # Change the respective zeros to ones
            ba_add_tensor = torch.bernoulli(np.clip(chance_add_per_channel,0,1) * torch.ones(data_batch.shape)).to(data_batch.device) # adds  spikes
            data_batch = torch.logical_or(data_batch, ba_add_tensor).to(data_batch.dtype).to(data_batch.device)
            del ba_add_tensor
            torch.cuda.empty_cache()
            # Change the respective ones to zeros
            ba_substract_tensor = torch.bernoulli(np.clip((1-chance_delete_per_channel),0,1) * torch.ones(data_batch.shape)).to(data_batch.device) # deletes  spikes
            data_batch = torch.logical_and(data_batch, ba_substract_tensor).to(data_batch.dtype).to(data_batch.device)
            del ba_substract_tensor
            torch.cuda.empty_cache()

        # Noise: Mismatch
        if glob_args != None and glob_args.mismatch != 0.0:
            net.add_mismatch(glob_args.mismatch)
        
        # Initialize and run network
        net.init(data_batch, burnin) 
        t_sample = data_batch.shape[1] 

        for k in (range(burnin,t_sample)):  # Iterates over 300 time steps minus burnin_steps (i.e. 60)         
            if glob_args != None and glob_args.mismatch_forward != 0.0:
                weight_0, weight_1, weight_2, bias_0, bias_1, bias_2 = net.get_LIF_layers()
                net.add_mismatch(glob_args.mismatch_forward)
                            
            s, r, u = net.step(data_batch[:, k, :, :], glob_args) 

            if glob_args != None and glob_args.mismatch_forward != 0.0:
                net.reset_mismatch(weight_0, weight_1, weight_2, bias_0, bias_1, bias_2)

            loss_ = decolle_loss(s, r, u, target=target_batch[:,k,:], mask = loss_mask[:,k,:], sum_ = False) # Calculating smooth l1 loss for all layers -> torch function
            total_loss += tonp(torch.tensor(loss_))  # Converts losses of three layers into a numpy array
            loss_tv += sum(loss_)  # Adds sum of loss of all layers at the moment to loss_tv
            if online_update:  
                loss_tv.backward() 
                opt.step() 
                opt.zero_grad() 
                for i in range(len(net)): 
                    act_rate[i] += tonp(s[i].mean().data)/t_sample 
                loss_tv = torch.tensor(0.).to(device) 
        if not online_update: 
            loss_tv.backward() 
            opt.step() 
            opt.zero_grad() 
            for i in range(len(net)): 
                act_rate[i] += tonp(s[i].mean().data)/t_sample 
            loss_tv = torch.tensor(0.).to(device) 
        batch_iter +=1 
        if batches_per_epoch>0: 
            if batch_iter >= batches_per_epoch: break 
 
    total_loss /= t_sample 
    print('Loss {0}'.format(total_loss)) 
    print('Activity Rate {0}'.format(act_rate)) 
    return total_loss, act_rate 
 
def test(gen_test, decolle_loss, net, burnin, glob_args=None, print_error = True, debug = False): 
    net.eval() 
    if hasattr(net.LIF_layers[0], 'base_layer'): 
        dtype = net.LIF_layers[0].base_layer.weight.dtype 
    else: 
        dtype = net.LIF_layers[0].weight.dtype 

    # Noise: Mismatch
    if glob_args!=None and glob_args.mismatch != 0:
        net.add_mismatch(glob_args.mismatch)
    
    with torch.no_grad(): 
        device = net.get_input_layer_device() 
        iter_data_labels = iter(gen_test) 
        test_res = [] 
        test_labels = [] 
        test_loss = np.zeros([decolle_loss.num_losses]) 
        batch_num = 0

        initialised = False

        if glob_args != None and glob_args.save_voltage != None:
            u_save = torch.Tensor()

        for data_batch, target_batch in tqdm.tqdm(iter_data_labels, desc='Testing'): 
            if initialised == False and glob_args != None and glob_args.save_voltage != None:
                already_evaluated = np.full((target_batch.shape[2]), False, dtype=bool)
                initialised = True
            
            if glob_args != None and glob_args.sam != None:
                spike_list = [torch.Tensor().to(device=glob_args.device), torch.Tensor().to(device=glob_args.device), torch.Tensor().to(device=glob_args.device)]
                delta_t = [torch.Tensor().to(device=glob_args.device), torch.Tensor().to(device=glob_args.device), torch.Tensor().to(device=glob_args.device)]
                sam_gif = [torch.Tensor().to(device=glob_args.device), torch.Tensor().to(device=glob_args.device), torch.Tensor().to(device=glob_args.device)]

            data_batch = torch.tensor(data_batch).type(dtype).to(device) 
            target_batch = torch.tensor(target_batch).type(dtype).to(device) 

            batch_size = data_batch.shape[0] 

            if glob_args!=None and glob_args.gif_save_dir != None:
                print_to_gif(data_batch, target_batch, dtype, device, glob_args, batch_num, "no_noise")

            # Noise: Hot Pixels
            if glob_args!=None and glob_args.hot_pixels > 0.0:
                for i in range(batch_size):
                    len_x = data_batch.shape[3]
                    len_y = data_batch.shape[4]
                    num_pixels = len_x * len_y
                    num_hot_pixels = int(num_pixels * glob_args.hot_pixels // 100)
                    pixels = [[i,j] for i in range(0,len_x) for j in range(len_y)]
                    hot_pixels = random.sample(pixels, num_hot_pixels)
                    for a in range(len(hot_pixels)):
                        channel = random.randint(0,1)
                        data_batch[i,:,channel, hot_pixels[a][0], hot_pixels[a][1]] = 1.0  # setting all values of one pixel in one sample to 1.0 for one polarity
                if glob_args.gif_save_dir != None:
                    print_to_gif(data_batch, target_batch, dtype, device, glob_args, batch_num, "hot_pixel_"+str(glob_args.hot_pixels))

            # Noise: Background Activity
            if glob_args!=None and glob_args.ba_noise > 0.0:
                for i in range(batch_size):
                    len_x = data_batch.shape[3]
                    len_y = data_batch.shape[4]
                    pixels = [[k,i,j] for i in range(0,len_x) for j in range(len_y) for k in range(2)]
                    for b in range(data_batch.shape[1]):
                        num_ba_pixels = np.random.poisson(glob_args.ba_noise)
                        ba_pixels = random.sample(pixels, num_ba_pixels)
                        for a in range(len(ba_pixels)):
                            data_batch[i,b,ba_pixels[a][0], ba_pixels[a][1], ba_pixels[a][2]] = 1.0
                if glob_args.gif_save_dir != None:
                    print_to_gif(data_batch, target_batch, dtype, device, glob_args, batch_num, "ba_noise_"+str(glob_args.ba_noise))

            # Noise: Background Activity (more efficient alternative implementation)
            if glob_args!=None and glob_args.ba_noise_torch > 0.0:
                for timestep in range(data_batch.shape[1]):
                    num_ba_pixels = np.random.poisson(glob_args.ba_noise_torch)
                    device = data_batch.device
                    dtype = data_batch.dtype
                    chance_add_per_channel = (num_ba_pixels) / (data_batch[:,timestep,:,:,:].shape[1] * data_batch[:,timestep,:,:,:].shape[2] * data_batch[:,timestep,:,:,:].shape[3])
                    ba_add_tensor = torch.bernoulli(np.clip(chance_add_per_channel,0,1) * torch.ones(data_batch[:,timestep,:,:,:].shape)).to(device) # adds  spikes
                    data_batch[:,timestep,:,:,:] = torch.logical_or(data_batch[:,timestep,:,:,:], ba_add_tensor).to(dtype).to(device)
            
            timesteps = data_batch.shape[1] 
            nclasses = target_batch.shape[2] 
            r_cum = np.zeros((decolle_loss.num_losses, timesteps-burnin, batch_size, nclasses)) 
 
            net.init(data_batch, burnin) 

            # If wanted: generate and return SAM gif
            if glob_args != None and glob_args.sam != None:
                if torch.argmax(target_batch[0,200,:]) == glob_args.sam:
                    for k in (range(burnin,timesteps)):
                        s, r, u = net.step(data_batch[:, k, :, :], glob_args, k, batch_num)
                        if len(s) == 3:
                            spike_list, delta_t, sam_gif = sam(spike_list, delta_t, sam_gif, s, 0.5, glob_args)       
                    return sam_gif               
            else:
                if glob_args != None and glob_args.save_voltage != None and already_evaluated[torch.argmax(target_batch[0,200,:])] == False:
                    for k in (range(burnin,timesteps)): 
                        s, r, u = net.step(data_batch[:, k, :, :], glob_args, k, batch_num)
                        torch.cuda.empty_cache()
                        if len(u) == 3:
                            u_save = torch.cat((u_save.to(u[glob_args.save_voltage].device), torch.flatten(u[glob_args.save_voltage])))
                            torch.cuda.empty_cache()
                    already_evaluated[torch.argmax(target_batch[0,200,:])] = True
                elif glob_args == None or glob_args.save_voltage == None:
                    for k in (range(burnin,timesteps)): 
                        s, r, u = net.step(data_batch[:, k, :, :], glob_args, k, batch_num) # Calculates s,r,u layerwise for batch r is random readout (calculated from s)
                        test_loss_tv = decolle_loss(s,r,u, target=target_batch[:,k], sum_ = False) # Layerwise (only ONE value per layer) Loss of batch at current time k calculated from r and target in L1SmoothLoss
                        test_loss += [tonp(x) for x in test_loss_tv] # Sum of all timestamps's test_loss_tv 
                        for l,n in enumerate(decolle_loss.loss_layer): 
                            r_cum[l,k-burnin,:,:] += tonp(sigmoid(r[n])) # Cumulates all r curved by torch.sigmoid in one batch over timesteps and loss layer
                    test_res.append(prediction_mostcommon(r_cum))  # Returns prediction given by summed random readout function for every layer in every sample
                    test_labels += tonp(target_batch).sum(1).argmax(axis=-1).tolist() 
                else:
                    if len(np.nonzero(already_evaluated)[0]) == target_batch.shape[2]:
                        return u_save.cpu().numpy()

            batch_num +=1

        test_acc  = accuracy(np.column_stack(test_res), np.column_stack(test_labels)) # returns acc layerwise |Test_res is array [#batches][#layer][batch_size]
        test_loss /= len(gen_test) 
        if print_error: 
            print(' '.join(['Error Rate L{0} {1:1.3}'.format(j, 1-v) for j, v in enumerate(test_acc)])) 
    if debug: 
        return test_loss, test_acc, s, r, u 
    else: 
        return test_loss, test_acc 
 
def sam(spike_list, delta_t, sam_gif, s, gamma = 0.5, glob_args=None):
    for layer in range(3):
        spike_list[layer] = torch.cat((spike_list[layer],s[layer]))   # Append current spikes to spike list

        # Calculate delta t tensor for all past spikes 
        delta_t[layer] = torch.cat((delta_t[layer], -1*torch.ones(s[layer].shape).to(device=glob_args.device)))
        delta_t[layer] = delta_t[layer] + torch.ones(delta_t[layer].shape).to(device=glob_args.device)

        # Calculate weight for current spikes
        weights = spike_list[layer] * torch.exp(gamma * (-1) * delta_t[layer])
        weight = torch.sum(weights, dim=0)

        # Calculate weighted activation
        weighted_activation = weight * torch.sum(s[layer], dim=0)

        # Sum weighted activation over channels
        sam_value = torch.sum(weighted_activation, dim=0)

        # Processing for plot taken from original code
        sam_value = sam_value - torch.min(sam_value)
        sam_img = sam_value / (torch.max(sam_value) +1e-3) 
        sam_img = torch.unsqueeze(sam_img, dim=0)

        # Save frame
        sam_gif[layer] = torch.cat((sam_gif[layer], sam_img))

    return spike_list, delta_t, sam_gif 

def accuracy(outputs, targets, one_hot = True): 
    if type(targets) is torch.tensor: 
        targets = tonp(targets) 
 
 
    return [np.mean(o==targets) for o in outputs] 
 
def prediction_mostcommon(outputs): 
    maxs = outputs.argmax(axis=-1) 
    res = [] 
    for m in maxs: 
        most_common_out = [] 
        for i in range(m.shape[1]): 
#            idx = m[:,i]!=target.shape[-1] #This is to prevent classifying the silence states 
            most_common_out.append(Counter(m[:, i]).most_common(1)[0][0]) 
        res.append(most_common_out) 
    return res 
 
def save_checkpoint(epoch, checkpoint_dir, net, opt): 
    if not os.path.exists(checkpoint_dir): 
        os.makedirs(checkpoint_dir) 
    torch.save({ 
        'epoch'               : epoch, 
        'model_state_dict'    : net.state_dict(), 
        'optimizer_state_dict': opt.state_dict(), 
        }, os.path.join(checkpoint_dir, 'epoch{:05}.tar'.format(epoch))) 

def write_stats(epoch, test_acc, test_loss, writer): 
    for i, [l, a] in enumerate(zip(test_loss, test_acc)): 
        writer.add_scalar('/test_loss/layer{}'.format(i), l, epoch) 
        writer.add_scalar('/test_acc/layer{}'.format(i), a, epoch) 

def get_output_shape(input_shape, kernel_size=[3,3], stride = [1,1], padding=[1,1], dilation=[0,0]): 
    if not hasattr(kernel_size, '__len__'): 
        kernel_size = [kernel_size, kernel_size] 
    if not hasattr(stride, '__len__'): 
        stride = [stride, stride] 
    if not hasattr(padding, '__len__'): 
        padding = [padding, padding] 
    if not hasattr(dilation, '__len__'): 
        dilation = [dilation, dilation] 
    im_height = input_shape[-2] 
    im_width = input_shape[-1] 
    height = int((im_height + 2 * padding[0] - dilation[0] * 
                  (kernel_size[0] - 1) - 1) // stride[0] + 1) 
    width = int((im_width + 2 * padding[1] - dilation[1] * 
                  (kernel_size[1] - 1) - 1) // stride[1] + 1) 
    return [height, width] 

class DictMultiOpt(object): 
    def __init__(self, params): 
        self.params = params 
    def __getitem__(self, key): 
        p = [] 
        for par in self.params: 
            p.append(par[key]) 
        return p 
    def __setitem__(self, key, values): 
        for i, par in enumerate(self.params): 
            par[key] = values[i] 

class MultiOpt(object): 
    def __init__(self, *opts): 
        self.optimizers = opts 
        self.multioptparam = DictMultiOpt([opt.param_groups[-1] for opt in self.optimizers]) 
 
    def zero_grad(self): 
        for opt in self.optimizers: 
            opt.zero_grad() 
 
    def step(self): 
        for opt in self.optimizers: 
            opt.step() 
     
    def __getstate__(self): 
        p = [] 
        for opt in self.optimizers: 
            p.append(opt.__getstate__()) 
        return p 
     
    def state_dict(self): 
        return self.__getstate__() 
     
    def load_state_dicts(self, state_dicts): 
        # need to load state_dict of each optimizer 
        for i,op in enumerate(self.optimizers): 
            op.load_state_dict(state_dicts[i]) 
 
    def __iter__(self): 
        for opt in self.optimizers: 
            yield opt 
     
    @property 
    def param_groups(self): 
        return [self.multioptparam] 
