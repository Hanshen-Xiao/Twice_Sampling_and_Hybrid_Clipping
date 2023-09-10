import torch
import torch.nn as nn
import torch.nn.functional as F
# from opacus.grad_sample import register_grad_sampler
import torch.nn.init as init

# we_store_mean_and_var = False
# class smart_batchnorm(nn.BatchNorm2d):
        
#     def __init__(self, 
#                  num_features, 
#                  eps = 1e-5, 
#                  momentum = 0.1,
#                  affine = True, 
#                  track_running_stats = True):
#         super(smart_batchnorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        
#         # self.mean_holder = None
#         # self.var_holder = None
        
#         # if affine:
#         #     ''' trainable para. '''
#         #     self.weight = nn.Parameter( torch.ones(num_features, requires_grad=True) )
#         #     self.bias = nn.Parameter( torch.zeros(num_features, requires_grad=True) )

#     def forward(self, input):
#         # return input
#         # return nn.BatchNorm2d( int(input.shape[1]), affine = False ).to(input.device)(input)
#         # return nn.GroupNorm(4, int(input.shape[1]), affine = False)(input)
        
#         global we_store_mean_and_var
#         self._check_input_dim(input)
#         if we_store_mean_and_var:
#             # print(f'==> pre computing mean and var for: {id(self)}')
            
#             ''' use pub data to compute mean and var stored for future use'''
#             # with torch.no_grad():
#             #     self.mean_holder = torch.clone(input.mean([0, 2, 3]).detach())
#             #     self.var_holder = torch.clone(input.var([0, 2, 3], unbiased=False).detach())

#             self.mean_holder = input.mean([0, 2, 3])
#             self.var_holder = input.var([0, 2, 3], unbiased=False)
            
#             mean = self.mean_holder
#             var = self.var_holder
            
#         else:
#             # print(f'==> not pre computing mean and var for: {id(self)}')
            
#             # exponential_average_factor = 0.0
#             # if self.training and self.track_running_stats:
#             #     if self.num_batches_tracked is not None:
#             #         self.num_batches_tracked += 1
#             #         if self.momentum is None:  # use cumulative moving average
#             #             exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#             #         else:  # use exponential moving average
#             #             exponential_average_factor = self.momentum          

#             if self.training:
#                 ''' (Normal training) in DP mode, compute mean and var using batched private data is not allowed '''
#                 # mean = input.mean([0, 2, 3])
#                 # # use biased var in train
#                 # var = input.var([0, 2, 3], unbiased=False)
                
#                 ''' (DP training) instead, using mean and stored var from previous computation '''
#                 if self.mean_holder is None:
#                     raise ValueError('implementation error: mean_holder is None')

#                 ''' using mean and var computed by previous computation '''
#                 mean = self.mean_holder 
#                 var =  self.var_holder 
                
#                 # n = input.numel() / input.size(1)
#                 # with torch.no_grad():
#                 #     self.running_mean = exponential_average_factor * mean\
#                 #         + (1 - exponential_average_factor) * self.running_mean
#                 #     # update running_var with unbiased var
#                 #     self.running_var = exponential_average_factor * var * n / (n - 1)\
#                 #         + (1 - exponential_average_factor) * self.running_var
#             else:
#                 ''' so, which mean and var should be used ?'''
#                 mean = input.mean([0, 2, 3])
#                 # use biased var in train
#                 var = input.var([0, 2, 3], unbiased=False)
                
#                 ''' using running mean and var '''
#                 # mean = self.running_mean
#                 # var = self.running_var
                
#         normalized_input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
#         if self.affine:
#             self.x = torch.clone(normalized_input.detach())
#             self.batch_num = int(input.shape[0])
#             output = normalized_input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
#             return output
#         else:
#             return normalized_input
    
#     @staticmethod
#     def set_store_mean_and_var(store_mean_and_var):
#         global we_store_mean_and_var
#         we_store_mean_and_var = store_mean_and_var
    
# ''' trainable para. '''
# @register_grad_sampler(smart_batchnorm)
# def compute_smart_batchnorm_grad_sample(
#                                         layer: nn.Linear, 
#                                         activations: torch.Tensor, 
#                                         backprops: torch.Tensor
#                                         ):
#     """
#     Computes per sample gradients for ``smart_batchnorm`` layer
#     Args:
#         layer: Layer
#         activations: Activations
#         backprops: Backpropagations
#     """
#     batch_num = layer.batch_num
    
#     gs = torch.einsum("ni...,ni...->ni", backprops, layer.x) #* batch_num
#     ret = {layer.weight: gs}
#     if layer.bias is not None:
#         ret[layer.bias] = torch.einsum("ni...->ni", backprops) #* batch_num
        
#     return ret


class smart_batchnorm(nn.BatchNorm2d):
        
    def __init__(self, 
                 num_features, 
                 eps = 1e-5, 
                 momentum = 0.1,
                 affine = False, 
                 track_running_stats = False,
                 out_type = None):
        super(smart_batchnorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        self.gn = nn.GroupNorm(min(4, num_features), num_features, affine =  False)
        # self.out_type = out_type

    def forward(self, x):
        # gn = self.gn(x.tensor)
        # return self.out_type(gn)
        # print(f'==> shape of x: {x.shape}, num of features: {self.num_features}')
        return self.gn(x)
        
        # ''' bn '''
        # self._check_input_dim(x)
        # mean = x.mean([0, 2, 3])
        # var = x.var([0, 2, 3], unbiased=False)
        # normalized_input = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        # if self.affine:
        #     return  normalized_input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        # else:
        #     return normalized_input

        ''' layer norm '''

        # mean = x.mean([1, 2, 3], keepdims=True)
        # var = x.var([1, 2, 3], keepdims = True, unbiased=False)
        # normalized_input = (x - mean) / (torch.sqrt(var + self.eps))

        # mean = torch.mean(x, dim = [1,2,3], keepdim=True)
        # var = torch.var(x, dim = [1,2,3], keepdim=True, unbiased=False)
        # normalized_input = (x - mean) / (torch.sqrt(var + self.eps))
        # return normalized_input

        # ''' instance norm, seems like not working '''
        # mean = x.mean([2, 3], keepdims=True)
        # var = x.var([2, 3], keepdims = True, unbiased=False)
        # normalized_input = (x - mean) / (torch.sqrt(var + self.eps))
        # return normalized_input

        # ''' channel norm '''
        # mean = x.mean([1,], keepdims=True)
        # var = x.var([1,], keepdims = True, unbiased=False)
        # normalized_input = (x - mean) / (torch.sqrt(var + self.eps))
        # return normalized_input

    
class Conv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    super().__init__(in_channels, out_channels, kernel_size, **kwargs)
  def forward(self, x):        
    weight = self.weight
    weight_mean = weight.mean(dim=(1,2,3), keepdim=True)
    std = weight.std(dim=(1,2,3), keepdim=True) + 1e-6
    weight = (weight - weight_mean)/ std / (weight.numel() / weight.size(0))**0.5
    return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
