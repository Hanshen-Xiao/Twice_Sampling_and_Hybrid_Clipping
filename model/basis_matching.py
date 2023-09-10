import torch
import torch.nn as nn
import numpy as np
import math

# package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad


def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


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


def clip_column(tsr, clip=1.0, p=2, inplace=True):
    if (inplace):
        avg_norm = inplace_clipping(tsr, torch.tensor(clip).cuda(tsr.device), p)
        return avg_norm
    else:
        norms = torch.norm(tsr, dim=1, p=p)
        #print(torch.median(norms).item(), torch.quantile(norms, 0.75).item(), torch.max(norms).item())
        scale = torch.clamp(clip / norms, max=1.0)
        avg_norm = torch.mean(norms).item()
        return avg_norm, tsr * scale.view(-1, 1)

def clamp_tensor(tsr, clamp_val, inplace=True):
    if inplace:
        tsr.clamp_(min = -clamp_val, max = clamp_val)
    else:
        return torch.clamp(tsr, min = -clamp_val, max = clamp_val)


def coordinate_sample(tsr, sample_rate, inplace=True):
    mask = (torch.rand_like(tsr).cuda(tsr.device) <= sample_rate)
    if inplace:
        tsr *= mask
    else:
        return tsr * mask

#@torch.jit.script
def inplace_clipping(matrix, clip, p):
    n, m = matrix.shape
    avg_norm = 0
    for i in range(n):
        # Normalize the i'th row
        col = matrix[i:i + 1, :]
        #col_norm = torch.sqrt(torch.sum(col ** 2))
        col_norm = (torch.sum(abs(col) **(p)))**(1/p)  #Hanshen 
        avg_norm += col_norm.item()
        if (col_norm > clip):
            col /= (col_norm / clip)
    return avg_norm / n


def check_approx_error(L, target):
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))
    if (target.item() == 0):
        return -1
    return error.item() / target.item()


def get_bases(pub_grad, num_bases, power_iter=1, logging=False):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, num_p)
    L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases), device=pub_grad.device)
    for i in range(power_iter):
        R = torch.matmul(pub_grad, L)  # n x k
        L = torch.matmul(pub_grad.T, R)  # p x k
        orthogonalize(L)
    # error_rate = check_approx_error(L, pub_grad)
    error_rate = 0
    return L, num_bases, error_rate

def get_bases_mul(pub_grad, num_bases_list, power_iter=1, logging=False):
    residual_grad = pub_grad
    selected_bases, new_num_bases_list = [], []
    for i, num_bases in enumerate(num_bases_list):
        L, new_num_bases, err = get_bases(residual_grad, num_bases, power_iter, logging)

        encode = torch.matmul(residual_grad, L)  # n x k
        decode = torch.matmul(encode, L.T)

        residual_grad -= decode
        new_num_bases_list.append(new_num_bases)
        selected_bases.append(L)

    selected_bases = torch.cat(selected_bases, dim=1)
    error_rate = check_approx_error(selected_bases, pub_grad)
    return selected_bases, new_num_bases_list, error_rate



def clip_column_mul(tsr, num_bases_space, clip, p):
    offset = 0
    for i, num_bases in enumerate(num_bases_space):
        avg_norm, tsr[:, offset:offset + num_bases] = clip_column(tsr[:, offset:offset + num_bases], clip[i], p[i], inplace=False)
        offset += num_bases


def clip_column_mul_adaptive(tsr, num_bases_space, clip, p, factor):
    offset = 0
    for i, num_bases in enumerate(num_bases_space):
        norms = torch.norm(tsr[:, offset:offset + num_bases], dim=1, p=p[i])
        clip[i] = torch.mean(norms).item() / factor
        scale = torch.clamp(clip[i] / norms, max=1.0)
        tsr[:, offset:offset + num_bases] *= scale.view(-1, 1)
        offset += num_bases



def clamp_tensor_mul(tsr, num_bases_space, clip, ratio):
    offset = 0
    #print(tsr.shape, num_bases_space, clip, ratio)
    for i, num_bases in enumerate(num_bases_space):
        clamp_tensor(tsr[:, offset:offset + num_bases], clip[i] * ratio[i], inplace=True)
        offset += num_bases


# transform a list of embedding (list based on param separation) into a single tensor with different orders
def transform(tsr, num_bases_list):
    if (len(tsr.shape) > 1):
        bs = tsr.shape[0]
    else:
        bs = 1
    tsr = tsr.view(bs, -1)

    param_num, space_num = len(num_bases_list), len(num_bases_list[0])
    new_embedding_list = [[] for i in range(space_num)]
    num_bases_space = [0] * space_num
    offset = 0
    #print("num_base", num_bases_list, tsr.shape)
    for i in range(param_num):
        for j in range(space_num):
            num_bases = num_bases_list[i][j]
            num_bases_space[j] += num_bases
            new_embedding_list[j].append(tsr[:, offset: offset + num_bases])
            offset += num_bases

    embedding_cat_list = []
    for embedding_space in new_embedding_list:
        embedding_cat_list.append(torch.cat(embedding_space, dim = 1))
    del new_embedding_list

    return torch.cat(embedding_cat_list, dim = 1), num_bases_space

def untransform(tsr, num_bases_list):
    if (len(tsr.shape) > 1):
        bs = tsr.shape[0]
    else:
        bs = 1
    tsr = tsr.view(bs, -1)

    param_num, space_num = len(num_bases_list), len(num_bases_list[0])
    new_embedding_list = [[] for i in range(param_num)]
    offset = 0
    for j in range(space_num):
        for i in range(param_num):
            num_bases = num_bases_list[i][j]
            new_embedding_list[i].append(tsr[:, offset: offset + num_bases])
            offset += num_bases

    embedding_cat_list = []
    for embedding_space in new_embedding_list:
        embedding_cat_list.append(torch.cat(embedding_space, dim = 1))
    del new_embedding_list
    return torch.cat(embedding_cat_list, dim = 1)




class GEP(nn.Module):

    def __init__(self, num_bases, batch_size, clip, ratio, clip_p, power_iter=1, 
            cor_sample=False, cor_sample_rate=1, adp_clip = True, batch_select=1):
        super(GEP, self).__init__()

        self.num_bases = num_bases
        self.num_bases_acc = num_bases
        self.clip = clip
        self.ratio = ratio
        self.clip_p = clip_p

        self.power_iter = power_iter
        self.batch_size = batch_size
        self.approx_error = {}
        self.batch_select = batch_select
        
        self.coordinate_sample = cor_sample
        self.cor_sample_rate = cor_sample_rate
        self.adaptive_clipping = adp_clip
        self.multi_aug = False
    
    def assign_multiaug_par(self, segment_bs, multi_times):
        self.multi_aug = True
        self.segment_bs = segment_bs
        self.multi_times = multi_times

    def get_approx_grad(self, embedding_input, transformed):
        if transformed:
            embedding = untransform(embedding_input, self.num_bases_list)
        else:
            embedding = embedding_input

        bases_list, num_bases_list, num_param_list = self.selected_bases_list, self.num_bases_list, self.num_param_list
        grad_list = []
        offset = 0
        if (len(embedding.shape) > 1):
            bs = embedding.shape[0]
        else:
            bs = 1
        embedding = embedding.view(bs, -1)

        for i, bases in enumerate(bases_list):
            num_bases = sum(num_bases_list[i])
            #print(bases.shape, embedding[:, offset:offset + num_bases].shape)

            grad = torch.matmul(embedding[:, offset:offset + num_bases].view(bs, -1), bases.T)
            if (bs > 1):
                grad_list.append(grad.view(bs, -1))
            else:
                grad_list.append(grad.view(-1))
            offset += num_bases
        if (bs > 1):
            return torch.cat(grad_list, dim=1)
        else:
            return torch.cat(grad_list)

    def get_anchor_gradients2(self, net, optimizer, loss_func):
        public_inputs, public_targets = self.public_inputs, self.public_targets
        batch_num = public_inputs.shape[0] / self.batch_select

        grad_list = {}
        for num in range(int(batch_num)):
            true_inputs = public_inputs[num * self.batch_select: (num + 1) * self.batch_select]
            true_targets = public_targets[num * self.batch_select: (num + 1) * self.batch_select]
            optimizer.zero_grad()
            true_outputs = net(true_inputs)
            loss = loss_func(true_outputs, true_targets)
            loss.backward()
            for name, param in net.named_parameters():
                if name not in grad_list:
                    grad_list[name] = param.grad.unsqueeze(0)
                else:
                    grad_list[name] = torch.cat((grad_list[name], param.grad.unsqueeze(0)), 0)

        cur_batch_grad_list = []
        for name, p in net.named_parameters():
            cur_batch_grad_list.append(grad_list[name].reshape(grad_list[name].shape[0], -1))
        return flatten_tensor(cur_batch_grad_list)

    def get_anchor_gradients(self, net, loss_func):
        public_inputs, public_targets = self.public_inputs, self.public_targets
        outputs = net(public_inputs)
        loss = loss_func(outputs, public_targets)
        loss.backward()
        cur_batch_grad_list = []
        for p in net.parameters():
            cur_batch_grad_list.append(p.grad_sample.reshape(p.grad_sample.shape[0], -1))
            del p.grad_sample
        return flatten_tensor(cur_batch_grad_list)

    def get_anchor_space(self, net, loss_func, logging=False):
        anchor_grads = self.get_anchor_gradients(net, loss_func)
        with torch.no_grad():
            num_param_list = self.num_param_list
            num_anchor_grads = anchor_grads.shape[0]
            num_group_p = len(num_param_list)

            selected_bases_list = []
            num_bases_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            num_bases_list = [ [int(b * p) for b in self.num_bases] for 
                            p in (sqrt_num_param_list / np.sum(sqrt_num_param_list))
                            ]

            total_p = 0
            offset = 0

            for i, num_param in enumerate(num_param_list):
                pub_grad = anchor_grads[:, offset:offset + num_param]
                offset += num_param

                num_bases = num_bases_list[i]

                selected_bases, num_bases, pub_error = get_bases_mul(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases)

            self.selected_bases_list = selected_bases_list
            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
        #print("approx error is: ", self.approx_errors)
        del anchor_grads

    def forward(self, target_grad, logging=False):
        with torch.no_grad():
            num_param_list = self.num_param_list
            embedding_list = []

            offset = 0
            if (logging):
                print('group wise approx error')

            for i, num_param in enumerate(num_param_list):
                grad = target_grad[:, offset:offset + num_param]
                selected_bases = self.selected_bases_list[i]

                embedding = torch.matmul(grad, selected_bases)
                embedding_list.append(embedding)             
                offset += num_param

                if (logging):
                    cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                    cur_target = torch.mean(grad, dim=0)
                    cur_error = torch.sum(torch.square(cur_approx - cur_target)) / torch.sum(torch.square(cur_target))
                    print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%' % (
                    i, num_param, sum(self.num_bases_list[i]), 100 * cur_error.item()))
                    if (i in self.approx_error):
                        self.approx_error[i].append(cur_error.item())
                    else:
                        self.approx_error[i] = []
                        self.approx_error[i].append(cur_error.item())

            embedding_cat = torch.cat(embedding_list, dim = 1)
            no_reduction_approx = self.get_approx_grad(embedding_cat, transformed = False)
            residual_gradients = target_grad - no_reduction_approx

            transformed_embedding, self.num_bases_space = transform(embedding_cat, self.num_bases_list)

            if (logging):
                norms = torch.norm(transformed_embedding, dim=1)
                print('average norm of embedding: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())

            avg_res_norm = clip_column(residual_gradients, clip=self.clip[-1], p=self.clip_p[-1])  # inplace clipping to save memory
            if self.ratio[-1] < 1:
                clamp_tensor(residual_gradients, self.clip[-1] * self.ratio[-1])
            if self.coordinate_sample:
                coordinate_sample(residual_gradients, self.cor_sample_rate)
            clipped_residual_gradients = residual_gradients

            if self.adaptive_clipping:
                clip_column_mul_adaptive(transformed_embedding, self.num_bases_space, 
                    clip=self.clip, p=self.clip_p, factor=avg_res_norm / self.clip[-1])
            else:
                clip_column_mul(transformed_embedding, self.num_bases_space, 
                    clip=self.clip, p=self.clip_p)
            clamp_tensor_mul(transformed_embedding, self.num_bases_space, clip=self.clip, ratio=self.ratio)
            if self.coordinate_sample:
                coordinate_sample(transformed_embedding, self.cor_sample_rate)
            
            if (logging):
                norms = torch.norm(transformed_embedding, dim=1)
                print('average norm of clipped embedding: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
            avg_clipped_embedding = torch.sum(transformed_embedding, dim=0) / self.batch_size

            if (logging):
                norms = torch.norm(clipped_residual_gradients, dim=1)
                print('average norm of clipped residual gradients: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())

            avg_clipped_residual_gradients = torch.sum(clipped_residual_gradients, dim=0) / self.batch_size
            avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
            return avg_clipped_embedding.view(-1), avg_clipped_residual_gradients.view(-1), avg_target_grad.view(-1), self.num_bases_space

