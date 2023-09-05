#!/bin/python
#-----------------------------------------------------------------------------
# File Name : allconv_decolle.py
# Author: Emre Neftci
#
# Creation Date : Wed 07 Aug 2019 07:00:31 AM PDT
# Last Modified : Sept 5. 2023, T. Stadtmann
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from .base_model import *

class LenetDECOLLE(DECOLLEBase):
    def __init__(self,
                 input_shape,
                 Nhid=[1],
                 Mhid=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 alpha=[.9],
                 beta=[.85],
                 alpharp=[.65],
                 wrp=[1.0],
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 deltat=1000,
                 lc_ampl=[.5],
                 lif_layer_type = LIFLayer,
                 method='rtrl',
                 with_output_layer = False,
                 with_bias = True,
                 threshold=0):

        self.with_output_layer = with_output_layer
        self.num_layers = num_layers = num_conv_layers + num_mlp_layers
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers
        self.with_bias = with_bias
        self.threshold = threshold

        if Mhid is None:
            Mhid = []
        if self.with_output_layer:
            Mhid += [out_channels]
            self.num_layers += 1

        # If only one value provided, then it is duplicated for each layer
        if self.num_conv_layers>0:
            if len(kernel_size) == 1:   kernel_size = kernel_size * self.num_conv_layers
            if stride is None: stride=[1]
            if len(stride) == 1:        stride = stride * self.num_conv_layers
            if pool_size is None: pool_size = [1]
            if len(pool_size) == 1:     pool_size = pool_size * self.num_conv_layers

        if len(alpha) == 1:         alpha = alpha * self.num_layers
        self.alpha = alpha
        if len(alpharp) == 1:       alpharp = alpharp * self.num_layers
        self.alpharp = alpharp
        if not hasattr(wrp, '__len__'): wrp = [wrp]
        if len(wrp) == 1:       wrp = wrp * self.num_layers
        self.wrp = wrp
        if len(beta) == 1:          beta = beta * self.num_layers
        self.beta = beta

        if dropout == [] or dropout is None:    dropout = [1.0]
        if not hasattr(dropout, '__len__'): dropout = [dropout]
        if len(dropout) == 1:       dropout = dropout * self.num_layers
        self.dropout = dropout

        if Nhid is None:            Nhid = []
        self.Nhid = Nhid
        if Mhid is None:            Mhid = []
        self.Mhid = Mhid

        if hasattr(lif_layer_type, '__len__'): 
            self.lif_layer_type = lif_layer_type
        else:
            self.lif_layer_type = [lif_layer_type]*len(Nhid) + [lif_layer_type]*len(Mhid)

        self.deltat = deltat
        self.method = method
        if lc_ampl is not None:
            lc_ampl = [lc_ampl]*self.num_layers
        self.lc_ampl = lc_ampl

            

        self.out_channels = out_channels

        super(LenetDECOLLE, self).__init__()


        # THe following lists need to be nn.ModuleList in order for pytorch to properly load and save the state_dict
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_shape = input_shape


        #Compute number channels for convolutional and feedforward stacks.

        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]

        if num_conv_layers == 0: #No convolutional layer
            mlp_in = int(np.prod(self.input_shape))
        else:
            self.Nhid = [input_shape[0]] + self.Nhid
            conv_stack_output_shape = self.build_conv_stack(self.Nhid, feature_height, feature_width, pool_size, kernel_size, stride, out_channels)
            mlp_in = int(np.prod(conv_stack_output_shape))
            
        self.Mhid = [mlp_in] + self.Mhid

        mlp_stack_output_shape = self.build_mlp_stack(self.Mhid, out_channels)

        if self.with_output_layer:
            output_shape = self.build_output_layer(self.Mhid, out_channels)


    def build_conv_stack(self, Nhid, feature_height, feature_width, pool_size, kernel_size, stride, out_channels):
        output_shape = None
        padding = (np.array(kernel_size) - 1) // 2  
        for i in range(self.num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width], 
                kernel_size = kernel_size[i],
                stride = stride[i],
                padding = padding[i],
                dilation = 1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = nn.Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = self.lif_layer_type[i](base_layer,
                             alpha=self.alpha[i],
                             beta=self.beta[i],
                             alpharp=self.alpharp[i],
                             wrp=self.wrp[i],
                             deltat=self.deltat,
                             do_detach= True if self.method == 'rtrl' else False,
                             threshold = self.threshold)
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            if self.lc_ampl is not None:
                readout = nn.Linear(int(feature_height * feature_width * Nhid[i + 1]), out_channels, bias=self.with_bias)

                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, self.lc_ampl[i])
            else:
                readout = nn.Identity()
            self.readout_layers.append(readout)

            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()


            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.dropout_layers.append(dropout_layer)
        return (Nhid[-1],feature_height, feature_width)

    def build_mlp_stack(self, Mhid, out_channels): 
        output_shape = None

        for i in range(self.num_mlp_layers):
            base_layer = nn.Linear(Mhid[i], Mhid[i+1], self.with_bias)
            layer = self.lif_layer_type[i+self.num_conv_layers](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         wrp=self.wrp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            if self.lc_ampl is not None:
                readout = nn.Linear(Mhid[i+1], out_channels)
                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, self.lc_ampl[i])
            else:
                readout = nn.Identity()

            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
            output_shape = out_channels

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
        return (output_shape,)

    def build_output_layer(self, Mhid, out_channels):
        if self.with_output_layer:
            i=self.num_mlp_layers
            base_layer = nn.Linear(Mhid[i], out_channels)
            layer = self.lif_layer_type[-1](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         wrp=self.wrp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            readout = nn.Identity()
            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
            
            output_shape = out_channels

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
        return (output_shape,)

    def save_membrane_voltage(self, u_p, save_dir, layer):
        u_p_list = np.mean(u_p.clone().cpu().detach().numpy().flatten()).tolist()
        if layer==0: 
            membrane_voltage_0 = np.load(save_dir+'_layer_0.npy')
            membrane_voltage_0 = np.append(membrane_voltage_0, u_p_list)
            np.save(save_dir+'_layer_0.npy', membrane_voltage_0)
        elif layer==1:
            membrane_voltage_1 = np.load(save_dir+'_layer_1.npy')
            membrane_voltage_1 = np.append(membrane_voltage_1, u_p_list)
            np.save(save_dir+'_layer_1.npy', membrane_voltage_1)
        elif layer==2:
            membrane_voltage_2 = np.load(save_dir+'_layer_2.npy')
            membrane_voltage_2 = np.append(membrane_voltage_2, u_p_list)
            np.save(save_dir+'_layer_2.npy', membrane_voltage_2)

    def save_membrane_voltage_torch(self, u, save_dir, layer):
        if layer==0: 
            membrane_voltage_0 = torch.load(save_dir+'_layer_0.pt', map_location=u.device)
            membrane_voltage_0 = torch.cat((membrane_voltage_0, u))
            torch.save(membrane_voltage_0, save_dir+'_layer_0.pt')
        elif layer==1:
            membrane_voltage_1 = torch.load(save_dir+'_layer_1.pt', map_location=u.device)
            membrane_voltage_1 = torch.cat((membrane_voltage_1, u))
            torch.save(membrane_voltage_1, save_dir+'_layer_1.pt')
        elif layer==2:
            membrane_voltage_2 = torch.load(save_dir+'_layer_2.pt', map_location=u.device)
            membrane_voltage_2 = torch.cat((membrane_voltage_2, u))
            torch.save(membrane_voltage_2, save_dir+'_layer_2.pt')

    def add_thermal_noise(self, u_p, thermal_noise):
        dtype = u_p.dtype        
        device = u_p.get_device()

        gaussian_noise_tensor = torch.Tensor(np.random.normal(loc=0.0, scale=thermal_noise, size=u_p.shape)).type(dtype).to(device)
        
        return torch.add(u_p, gaussian_noise_tensor)

    def add_spike_loss(self, s_, spike_loss):
        loss_input = (1- spike_loss / 100) * s_
        zeroing = torch.bernoulli(loss_input)
        s_ *= zeroing

    def add_spike_add(self, s_, spike_add):
        add_input = spike_add/100 * torch.ones(s_.shape[:]).to(device=s_.device).to(dtype=s_.dtype)
        addition = torch.bernoulli(add_input)
        s_ = torch.logical_or(s_, addition)
        return s_

    def step(self, input, glob_args=None, k=0, batch_num=0, *args, **kwargs):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers):
            if i == self.num_conv_layers: 
                input = input.view(input.size(0), -1)
            s, u = lif(input, glob_args)   # calls forward function of the LIF layer easy accessible ; called in forward_call(*input, **kwargs)
            u_p = pool(u)       # Default from nn -> Hardly accessible || probably no changes in pooling
            
            if glob_args != None and glob_args.membrane_voltage_save_dir != None and k%100 == 0 and batch_num==0: #Save membrane voltage every 100 timesteps in first batch to reduce data
                self.save_membrane_voltage_torch(u, glob_args.membrane_voltage_save_dir, i)

            if i+1 == self.num_layers and self.with_output_layer:
                s_ = sigmoid(u_p)  # Default from nn -> Hardly accessible || only for output layer
                sd_ = u_p
            else:
                s_ = lif.sg_function(u_p)   # Calls sg_function of LIF layer
                sd_ = do(s_)                # Calls droput layer from nn -> Hardly accessible            
            r_ = ro(sd_.reshape(sd_.size(0), -1))

            s_out.append(s_) 
            r_out.append(r_)
            u_out.append(u_p)

            # Spike Loss (Spike loss happens on connections between layers)
            if glob_args != None and glob_args.spike_loss != 0.0:
                 self.add_spike_loss(s_, glob_args.spike_loss)

            if glob_args != None and glob_args.spike_add != 0.0:
                 s_ = self.add_spike_add(s_, glob_args.spike_add)

            input = s_.detach() if lif.do_detach else s_
            i+=1

        return s_out, r_out, u_out
