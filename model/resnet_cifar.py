import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models import norm_layer as nl
from functools import partial

print(sys.path)

smart_batchnorm = nl.smart_batchnorm
# smart_batchnorm = nn.BatchNorm2d




class special_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        std = weight.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        weight = (weight - weight_mean) / std / (weight.numel() / weight.size(0)) ** 0.5
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

conv_layer = partial(nn.Conv2d, bias = False)
# conv_layer = partial(special_Conv2d, bias = False)
''' 3 places are modified: base block, resblock, conv, final linear layer '''

''' the following is the resnet implementation '''
def _weights_init(m):
    # classname = m.__class__.__name__
    # #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, )
        self.bn1 = smart_batchnorm(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = smart_batchnorm(planes)
        self.bn3 = smart_batchnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     conv_layer(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                     smart_batchnorm(self.expansion * planes)
                )


    def forward(self, x):
        # ''' norm layer before activation '''
        # NLA = F.relu
        # out = NLA(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out = out + self.shortcut(x)
        # out = NLA(out)
        
        # NLA = F.elu
        # out = NLA(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out = out + self.shortcut(x)
        # out = NLA(out)

        # ''' norm layer after activation '''
        # NLA = F.elu
        # out = self.bn1(NLA(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out = out + self.shortcut(x)
        # out = NLA(out)
        
        ''' norm layer after activation good '''
        NLA = F.elu
        out = self.bn1(NLA(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.bn3( NLA(out) )

        # ''' norm layer after activation '''
        # NLA = F.elu
        # out = self.bn1(NLA(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out = out + self.shortcut(x)
        # out = self.bn3(out)

        # ''' norm layer after activation '''
        # NLA = F.elu
        # out = self.bn1(NLA(self.conv1(x)))
        # out = self.bn2(NLA(self.conv2(out)))
        # out = out + self.shortcut(x)
        # out = self.bn3( NLA(out) )
    
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = None):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv_layer(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = smart_batchnorm(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * 9, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # out = torch.tanh(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = F.adaptive_avg_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        
        # out = self.bn1( F.elu( self.conv1(x) ) )
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)

        out = self.bn1( F.elu( self.conv1(x) ) )
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)

        ''''''
        out = (out - out.mean(dim=1, keepdim=True)) / (out.std(dim=1, keepdim=True)+ 1e-6)
        # out = ( out - torch.mean(out) ) / ( torch.std(out)+ 1e-6 )
        
        return self.linear(out)



''' model candidates '''
def resnet14(num_class):
    return ResNet(BasicBlock, [2, 2, 2], num_class)

def resnet20(num_class):
    return ResNet(BasicBlock, [3, 3, 3], num_class)

def resnet22(num_class):
    return ResNet(BasicBlock, [3, 4, 3], num_class)

def resnet26(num_class):
    return ResNet(BasicBlock, [4, 4, 4], num_class)

def resnet32(num_class):
    return ResNet(BasicBlock, [5, 5, 5], num_class)

def resnet38(num_class):
    return ResNet(BasicBlock, [6, 6, 6], num_class)

def resnet44(num_class):
    return ResNet(BasicBlock, [7, 7, 7], num_class)

def resnet50(num_class):
    return ResNet(BasicBlock, [8, 8, 8], num_class)

def resnet56(num_class):
    return ResNet(BasicBlock, [9, 9, 9], num_class)

def test():
    net = resnet20()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ''' original model '''

# '''
# Properly implemented ResNet-s for CIFAR10 as described in paper [1].
# The implementation and structure of this file is hugely influenced by [2]
# which is implemented for ImageNet and doesn't have option A for identity.
# Moreover, most of the implementations on the web is copy-paste from
# torchvision's resnet and has wrong number of params.
# Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
# number of layers and parameters:
# name      | layers | params
# ResNet20  |    20  | 0.27M
# ResNet32  |    32  | 0.46M
# ResNet44  |    44  | 0.66M
# ResNet56  |    56  | 0.85M
# ResNet110 |   110  |  1.7M
# ResNet1202|  1202  | 19.4m
# which this implementation indeed has.
# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# If you use this implementation in you work, please don't forget to mention the
# author, Yerlan Idelbayev.
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init

# __all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

# def _weights_init(m):
#     classname = m.__class__.__name__
#     #print(classname)
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight)

# class LambdaLayer(nn.Module):
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd

#     def forward(self, x):
#         return self.lambd(x)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, option='A'):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = smart_batchnorm(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = smart_batchnorm(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             if option == 'A':
#                 """
#                 For CIFAR10 ResNet paper uses option A.
#                 """
#                 self.shortcut = LambdaLayer(lambda x:
#                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
#             elif option == 'B':
#                 self.shortcut = nn.Sequential(
#                      nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                      smart_batchnorm(self.expansion * planes)
#                 )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = smart_batchnorm(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64, num_classes)

#         self.apply(_weights_init)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# def resnet20():
#     return ResNet(BasicBlock, [3, 3, 3])


# def resnet32():
#     return ResNet(BasicBlock, [5, 5, 5])


# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])


# def resnet56():
#     return ResNet(BasicBlock, [9, 9, 9])


# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])


# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])


# def test(net):
#     import numpy as np
#     total_params = 0

#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.data.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


# if __name__ == "__main__":
#     for net_name in __all__:
#         if net_name.startswith('resnet'):
#             print(net_name)
#             test(globals()[net_name]())
#             print()
