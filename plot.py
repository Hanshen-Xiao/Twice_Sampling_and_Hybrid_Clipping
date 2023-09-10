import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import argparse
import csv
import random
import time
import numpy as np
import resnet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axis as ax

from opacus import PrivacyEngine
from opacus.utils import module_modification
from models import resnet20, resnet22, resnet26, GEP
from old_resnet import resnet20 as oldresnet20
from utils import get_data_loader, get_sigma, restore_param, sum_list_tensor, flatten_tensor, checkpoint, \
    adjust_learning_rate, multi_sample, avg_columns

# package for computing individual gradients
# from backpack import backpack, extend
# from backpack.extensions import BatchGrad

parser = argparse.ArgumentParser(description='Differentially Private learning with GEP')

## general arguments
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--resume', default=True, action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
parser.add_argument('--seed', default=2, type=int, help='random seed')
parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
parser.add_argument('--batchsize', default= 1000, type=int, help='batch size')
parser.add_argument('--n_epoch', default= 100, type=int, help='total number of epochs')
parser.add_argument('--lr', default= 2.5, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--momentum', default= 0.9, type=float, help='value of momentum')

## arguments for learning with differential privacy
parser.add_argument('--private', '-p', default=True, action='store_true', help='enable differential privacy')
parser.add_argument('--eps', default=20000, type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')

#parser.add_argument('--clip', default=[2, 0.9, 0.63, 0.45, 1], type=list, help='clipping threshold for gradient embedding')
parser.add_argument('--clip', default=[1, 1, 1, 1, 1], type=list, help='clipping threshold for gradient embedding')
parser.add_argument('--ratio', default=[0.25, 0.1, 0.1, 0.1, 0.1], type=list, help='infinite clipping ratio threshold for gradient embedding')
parser.add_argument('--clip_p', default=[2, 2, 2, 2, 2, 2], type=list, help='clipping p for gradient embedding')
parser.add_argument('--power_iter', default=1, type=int, help='number of power iterations')
parser.add_argument('--num_groups', default=10, type=int, help='number of parameters groups')
#parser.add_argument('--num_bases', default=[250, 500, 1000, 1500], type=list, help='dimension of anchor subspace')
parser.add_argument('--num_bases', default=[10], type=list, help='dimension of anchor subspace')


parser.add_argument('--adaptive_clip', default=True, action='store_true', help='enable adaptive clipping') # for testing only
parser.add_argument('--coordinate_sample', default=False, action='store_true', help='enable coordinate sampling')
parser.add_argument('--cor_sample_rate', default=1, type=float, help='sampling probability for coordinate sampling')
parser.add_argument('--multi_aug', default= True, action='store_true', help='repeatly augment a sample multiple times')
parser.add_argument('--multi_times', default= 6, type=int, help='how many times we augment each sample')
parser.add_argument('--segment_bs', default= 500, type=int, help='what is the batchsize for each segment')
# memory might not support multi_times * bs number of samples, thus we divide batch into segments of size segment_bs, splease select segment_bs such that it divides bs

parser.add_argument('--real_labels', action='store_true', help='use real labels for auxiliary dataset')
parser.add_argument('--aux_dataset', default='imagenet', type=str,
                    help='name of the public dataset, [cifar10, cifar100, imagenet]')
parser.add_argument('--aux_data_size', default= 2000, type=int, help='size of the auxiliary dataset')

use_cuda = True
gpu_device = torch.device('cuda:1')

#noise_multiplier = [6, 7.3, 7.5, 7.7, 2.3]
#noise_multiplier = [3.1, 3.9, 4.3, 4.4, 1.2]
#noise_multiplier = [7, 8.8, 9, 9.6, 2.6]
#noise_multiplier = [6.5, 8, 8.2, 8.7, 2.7]  #2000 6250/250 epoch
#noise_multiplier = [3.9, 4.8, 4.7, 5.3, 1.8]  #2000 2500/100 epoch
#noise_multiplier = [4, 5.2, 5.2, 5.2, 1.5]  #2000 2000/80 epoch
#noise_multiplier = [3.6, 4.4, 4.6, 5, 1.3]  #2000 2000/80 epoch

#noise_multiplier = [10, 12.4, 12, 12.9, 3.8]  #B2000 eps=2 1000/40 epoch
#noise_multiplier= [9.4000, 11.8000, 11.8000,12.8000, 3.8000]  #B2000 eps=2.5 1500/60 epoch
#noise_multiplier= [5.3, 7.8, 7.9, 9.2, 2.4]  #B2000 eps=4 1500/60 epoch

#noise_multiplier = [9.6, 11.8, 11.9, 3.7]  #B2000 eps=2 1000/40 epoch

#noise_multiplier = [5.3, 6.5, 6.5, 7, 1.9]  #B1000 eps=2 1000/40 epoch

#noise_multiplier = [3, 3.75, 4, 4.25, 1.5]  #B1000 eps=2 500/2000 iteration
#noise_multiplier = [3, 3.75, 3.75, 4, 1.5]  #B1000 eps=2.5 500/3000 iteration

#noise_multiplier = [3.2500, 4.2500, 4.2500, 4.5000, 1.7500] #0.333 eps=2.5 500/4000 iteration
#noise_multiplier = [3.500, 4.2500, 4.500, 5.0, 1.7500] #0.5 eps=2.5 500/4000 iteration
#noise_multiplier = [2.2500,2.7500,2.7500,3.0000,1.2500] #0.5 eps=4 500/4000 iteration

#noise_multiplier = [1.2500,1.7500,1.7500,2.25, 0.75] #0.3333 eps=8 500/5000 iteration

#noise_multiplier = [1.195, 1.195] #0.5 eps=2  500/2000 iteration
noise_multiplier = [1.05, 1.05] #0.5 eps=2  500/1000 iteration
#noise_multiplier = [3.500, 4.500, 4.500, 5, 2] #B1000 eps=2.5 500/5000 iteration



args = parser.parse_args()
print('eps=2')
assert args.dataset in ['cifar10', 'svhn']
assert args.aux_dataset in ['cifar10', 'cifar100', 'imagenet']
if (args.real_labels):
    assert args.aux_dataset == 'cifar10'

best_acc = 0
start_epoch = 0
batch_size = args.batchsize

if (args.seed != -1):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

print('==> Preparing data..')
## preparing data for training && testing
if (args.dataset == 'svhn'):  ## For SVHN, we concatenate training samples and extra samples to build the training set.
    trainloader, extraloader, testloader, n_training, n_test = get_data_loader('svhn', batchsize=args.batchsize)
    for train_samples, train_labels in trainloader:
        break
    for extra_samples, extra_labels in extraloader:
        break
    train_samples = torch.cat([train_samples, extra_samples], dim=0)
    train_labels = torch.cat([train_labels, extra_labels], dim=0)

else:
    trainloader, testloader, trainset, testset = get_data_loader('cifar10', batchsize=args.batchsize)
    n_training, n_test = len(trainset), len(testset)
    train_samples, train_labels = None, None
## preparing auxiliary data
num_public_examples = args.aux_data_size
if ('cifar' in args.aux_dataset):
    if (args.aux_dataset == 'cifar100'):
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    if (args.aux_dataset == 'cifar10'):
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    public_data_loader = torch.utils.data.DataLoader(testset, batch_size=num_public_examples, shuffle=False,
                                                     num_workers=2)  #
    for public_inputs, public_targets in public_data_loader:
        break
else:
    public_inputs = torch.load('imagenet_examples_2000')[:num_public_examples]
if (not args.real_labels):
    public_targets = torch.randint(high=10, size=(num_public_examples,))
public_inputs, public_targets = public_inputs.cuda(gpu_device), public_targets.cuda(gpu_device)
print('# of training examples: ', n_training, '# of testing examples: ', n_test, '# of auxiliary examples: ',
      num_public_examples)

print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP' % (args.eps, args.delta))
sampling_prob = args.batchsize / n_training
steps = int(args.n_epoch / sampling_prob)
sigma, eps = get_sigma(sampling_prob, steps, args.eps, args.delta, rgp=True)
# noise_multiplier0 = noise_multiplier1 = sigma

print('noise scale', noise_multiplier, 'privacy guarantee: ', eps)

print('\n==> Creating GEP class instance')
gep = GEP(args.num_bases, args.batchsize, args.clip, args.ratio, args.clip_p, args.power_iter,
          cor_sample=args.coordinate_sample, cor_sample_rate=args.cor_sample_rate,
          adp_clip=args.adaptive_clip).cuda(gpu_device)
gep.assign_multiaug_par(args.segment_bs, args.multi_times)
## attach auxiliary data to GEP instance
gep.public_inputs = public_inputs
gep.public_targets = public_targets

print('\n==> Creating ResNet20 model instance')
if (args.resume):
    try:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_file = './checkpoint/' + args.sess + '.ckpt'
        checkpoint = torch.load(checkpoint_file)
        net = resnet22(num_class=10)
        net.cuda(gpu_device)
        restore_param(net.state_dict(), checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
        approx_error = checkpoint['approx_error']
    except:
        print('resume from checkpoint failed')
else:
    net = resnet22(num_class=10)
    # net = oldresnet20()
    # net = resnet.__dict__["resnet20"]()
    # net = module_modification.convert_batchnorm_modules(net)
    net.cuda(gpu_device)

# net = extend(net)

net.gep = gep

num_params = 0
for p in net.parameters():
    num_params += p.numel()

print('total number of parameters: ', num_params / (10 ** 6), 'M')

if (args.private):
    loss_func = nn.CrossEntropyLoss(reduction='mean')
else:
    loss_func = nn.CrossEntropyLoss(reduction='mean')

# loss_func = extend(loss_func)

num_params = 0
np_list = []
for p in net.parameters():
    num_params += p.numel()
    np_list.append(p.numel())


def group_params(num_p, groups):
    assert groups >= 1

    p_per_group = num_p // groups
    num_param_list = [p_per_group] * (groups - 1)
    num_param_list = num_param_list + [num_p - sum(num_param_list)]
    return num_param_list


print('\n==> Dividing parameters in to %d groups' % args.num_groups)
gep.num_param_list = group_params(num_params, args.num_groups)

if args.coordinate_sample:
    args.lr /= args.cor_sample_rate
optimizer = optim.SGD(
    net.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)

privacy_engine = PrivacyEngine(net, batch_size=args.batchsize, sample_size=n_training, alphas=range(2, 32),
                               noise_multiplier=0, max_grad_norm=5)
privacy_engine.attach(optimizer)

space_num = len(args.clip)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    if (args.multi_aug):
        steps = n_training // args.segment_bs
        merge_steps = args.batchsize // args.segment_bs
    else:
        steps = n_training // args.batchsize
        merge_steps = 1

    if (train_samples == None and not args.multi_aug):  # using pytorch data loader for CIFAR10
        loader = iter(trainloader)
    else:  # manually sample minibatchs for SVHN
        sample_idxes = np.arange(n_training)
        np.random.shuffle(sample_idxes)

    for batch_idx in range(steps):
        if batch_idx % merge_steps == 0:
            acc_clipped_grad = None
            acc_residual = None

        if (args.dataset == 'svhn'):
            current_batch_idxes = sample_idxes[batch_idx * args.batchsize: (batch_idx + 1) * args.batchsize]
            inputs, targets = train_samples[current_batch_idxes], train_labels[current_batch_idxes]
        else:
            if (args.multi_aug):
                current_batch_idxes = sample_idxes[batch_idx * args.segment_bs: (batch_idx + 1) * args.segment_bs]
                inputs, targets = multi_sample(trainset, current_batch_idxes, args.multi_times)
            else:
                inputs, targets = next(loader)
        if use_cuda:
            inputs, targets = inputs.cuda(gpu_device), targets.cuda(gpu_device)
        optimizer.zero_grad()

        if (args.private):
            logging = batch_idx % 100 == 0
            ## compute anchor subspace
            if batch_idx % merge_steps == 0:
                net.gep.get_anchor_space(net, loss_func=loss_func, logging=logging)
            ## collect batch gradients
            batch_grad_list = []
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()

            # norms = None
            # for p in net.parameters():
            #    if norms is None:
            #        norms = [torch.norm(g) ** 2 for g in p.grad_sample]
            #    else:
            #        norms += [torch.norm(g) ** 2 for g in p.grad_sample]
            # print(min(norms), sum(norms) / len(norms), max(norms))

            for p in net.parameters():
                batch_grad_list.append(p.grad_sample.reshape(p.grad_sample.shape[0], -1))
                del p.grad_sample

            norms = torch.norm(flatten_tensor(batch_grad_list), dim = 1)
            #print("before flat: ", norms)

            ## compute gradient embeddings and residual gradients
            if args.multi_aug:
                flat_grad = avg_columns(flatten_tensor(batch_grad_list), args.multi_times)
                norms = torch.norm(flat_grad, dim = 1)
            else:
                flat_grad = flatten_tensor(batch_grad_list)
            return flat_grad
 
@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col

def check_approx_error(L, target):
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))
    if (target.item() == 0):
        return -1
    return error.item() / target.item()


def get_bases(pub_grad, num_bases, power_iter=1):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, num_p)
    L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases))
    L = L.cuda(gpu_device)
    for i in range(power_iter):
        R = torch.matmul(pub_grad, L)  # n x k
        L = torch.matmul(pub_grad.T, R)  # p x k
        orthogonalize(L)
    error_rate = check_approx_error(L, pub_grad)
    print(error_rate)
    return L, num_bases, error_rate
 
public_data0 = train(0).cuda(gpu_device)
public_data1 = train(1).cuda(gpu_device)
private_data0 = train(2).cuda(gpu_device)
private_data1 = train(3).cuda(gpu_device)

public_data = torch.cat((public_data0, public_data1), dim = 0)
private_data = torch.cat((private_data0, private_data1), dim = 0)
print(public_data.shape)


sample_norm = torch.norm(public_data, dim = 1)
public_data = public_data / sample_norm.view(-1, 1)
sample_norm = torch.norm(private_data, dim = 1)
private_data = private_data / sample_norm.view(-1, 1)


plt.figure(figsize = (10, 10))
plt.grid(linestyle='dashed')
plt.tick_params(labelsize=40)

if False:
    mean_col = torch.mean(torch.abs(private_data), dim = 0)
    private_data -= mean_col
    var_col = torch.sum(private_data ** 2, dim = 0)
    #S, idx = torch.sort(var_col / mean_col, descending=True)
    plt.plot((var_col / torch.abs(mean_col)).cpu())
    plt.xlabel('Coordinate Index', fontdict = {'size': 40})
    plt.ylabel('Coefficient of Variation', fontdict = {'size': 40})
    #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    #ax.xaxis.get_offset_text().set_fontsize(40)
    plt.xticks([0, 300000])
    plt.tight_layout()
    plt.show()

if False:
    true_grad = torch.mean(private_data, dim = 0)
    S, idx = torch.sort(torch.abs(true_grad), descending=True)
    norm_col = torch.norm(private_data, dim = 0) ** 2
    norm_col = norm_col[idx]
    for i, val in enumerate(norm_col):
        if i >= 1:
            norm_col[i] += norm_col[i - 1]
    plt.plot([i * 100 / norm_col.shape[0] for i in range(norm_col.shape[0])], (1 - norm_col / 1000).cpu(), 
        linewidth=4, linestyle='dashed')
    plt.xlabel('Quantile Ratio %', fontdict = {'size': 40})
    plt.ylabel('Norm of Residue', fontdict = {'size': 40})
    plt.tight_layout()
    plt.show()

if False:
    #L, num_bases, err = get_bases(public_data, 400, power_iter = 10)
    #proj = torch.matmul(public_data, L)
    #norms = torch.norm(proj, dim = 0)
    #S, idx = torch.sort(norms, descending = True)
    #print(norms[idx])

    L, num_bases, err = get_bases(public_data, 1000, power_iter = 1)
    proj = torch.matmul(private_data, L)
    residual = private_data
    norm_col = []

    for i in range(1000):
        a = proj.T[i].unsqueeze(0)
        b = L.T[i].unsqueeze(0)
        #print(a.shape, b.shape)
        residual -= torch.matmul(a.T.cuda(gpu_device), b.cuda(gpu_device))
        norm_col.append((torch.mean(torch.norm(residual, dim = 1) ** 2) ** 0.5).item())
    plt.plot(norm_col, linewidth=4, linestyle='dashed')
    plt.xlabel('Principle Space Dimension', fontdict = {'size': 40})
    plt.ylabel('Norm of Residue', fontdict = {'size': 40})
    #plt.yticks([0.2, 0.25, 0.3])
    plt.xticks([0, 500, 1000])
    plt.tight_layout()
    plt.show()

if True:
    #U, S, L = torch.linalg.svd(public_data)
    L, num_bases, err = get_bases(public_data, 1000, power_iter = 1)
    proj = torch.matmul(private_data, L)
    residual = private_data
    mean = []
    var = []

    for i in range(1000):
        a = proj.T[i].unsqueeze(0)
        b = L.T[i].unsqueeze(0)
        residual -= torch.matmul(a.T, b)
        norm_2 = torch.norm(residual, dim = 1)
        mean_norm = torch.mean(norm_2)
        mean.append(mean_norm.item())
        var.append((torch.mean( (norm_2 - mean_norm) ** 2 ) ** 0.5).item())
    print(var)
    #plt.scatter(range(len(mean)), mean, label = 'mean', linewidth=5)
    plt.plot(mean, linewidth=4, label = 'mean', linestyle='dashed')
    plt.plot(var, linewidth=4, label = 'std', linestyle='dashed')
    plt.legend(fontsize="40")
    plt.yticks([0, 0.08,  0.2, 0.5, 1])
    plt.xticks([0, 500, 1000])
    plt.xlabel('Principle space Dimension', fontdict = {'size': 40})
    plt.tight_layout()
    plt.show()


if False:
    l_inf, idx = torch.max(private_data, dim = 1)
    l_inf, idx = torch.sort(l_inf, descending=True)
    plt.plot(l_inf.cpu(), linewidth=4, linestyle='dashed')
    plt.xlabel('Sorted Sample Index', fontdict = {'size': 40})
    plt.ylabel('Infinity Norm of Samples', fontdict = {'size': 40})
    plt.xticks([0, 500, 1000])
    plt.tight_layout()
    plt.show()
 
 


