from __future__ import division
import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import time
from datetime import datetime, timezone

# Make the repository-level devkit importable regardless of the working
# directory used to launch this script.
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import models
import numpy as np
from devkit.sparse_ops import SparseConv,SparseLinear

from devkit.core import (
    LRScheduler,
    average_gradients,
    broadcast_params,
    cleanup_dist,
    conv_flops_counter_hook,
    get_device,
    get_sparsities_erdos_renyi,
    init_dist,
    normalize_erk_sparsity,
    reduce_tensor,
    summary,
    update_erk_sparsity,
)

from devkit.core import mean_var_group,get_layer_wise_dense_flops_params


from devkit.dataset.imagenet_dataset import ColorAugmentation
# Fine mixed N:M from dense net

parser = argparse.ArgumentParser(
    description='Search layer-wise N:M schemes on ImageNet')
parser.add_argument('--config', default='configs/config_resnet50_2:4.yaml')
parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int)
parser.add_argument(
    '--port', default=29500, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--target_sparsity', default=0.0, type=float)
parser.add_argument(
    '--target-metric',
    choices=('params', 'flops'),
    default='params',
    help='stop when the requested parameter or FLOPs reduction is reached',
)
parser.add_argument(
    '--erk-weight',
    type=float,
    default=None,
    help='ERK redundancy weight; defaults to 0.5 for params and 0.2 for FLOPs',
)
parser.add_argument(
    '--cost-weight',
    type=float,
    default=None,
    help='complexity weight; defaults to 0.5 for params and 0.8 for FLOPs',
)
parser.add_argument('--vote-ratio', type=float, default=0.75)
parser.add_argument('--model_dir', type=str)
parser.add_argument(
    '--scheme-output',
    type=str,
    default=None,
    help='output path for the searched layer-wise scheme',
)
parser.add_argument('--resume_from', default='', help='resume_from')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

args = None


# start_epoch_complexity_loss = 20
#state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
# global layer_sparse_vis_dict
    
    
# layer_sparse_vis_dict = {}

num_iters = 0


target_sparsity = 0.0
target_sparsity_erk = 0.0


decision_dict = {}

vote_ratio = 0.75


# hyper parameters study can be found in our paper
w1=0.5 # erk 
w2=0.5 # flops 

erk_sparsity_dict = {}


def configure_search_objective():
    global vote_ratio, w1, w2
    if not 0.0 < args.target_sparsity < 1.0:
        raise ValueError('--target_sparsity must be between 0 and 1 (exclusive).')
    if not 0.0 < args.vote_ratio <= 1.0:
        raise ValueError('--vote-ratio must satisfy 0 < ratio <= 1.')

    default_erk, default_cost = (
        (0.5, 0.5) if args.target_metric == 'params' else (0.2, 0.8)
    )
    w1 = default_erk if args.erk_weight is None else args.erk_weight
    w2 = default_cost if args.cost_weight is None else args.cost_weight
    if w1 < 0.0 or w2 < 0.0 or abs((w1 + w2) - 1.0) > 1e-8:
        raise ValueError('--erk-weight and --cost-weight must be non-negative and sum to 1.')
    vote_ratio = args.vote_ratio


def git_value(*arguments):
    try:
        result = subprocess.run(
            ['git'] + list(arguments),
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()

def main():
    global args, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("The config file is empty: {}".format(args.config))

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    configure_search_objective()
    rank, world_size = init_dist(
        backend='nccl', port=args.port)
    args.rank = rank
    args.world_size = world_size
    args.device = get_device()
    if rank == 0:
        mode = "distributed ({} GPUs)".format(world_size) if world_size > 1 else "single GPU"
        print("Running in {} mode on {}.".format(mode, args.device))
        print(
            "Search objective: {} reduction {:.4f}; ERK weight {:.2f}; cost weight {:.2f}; vote ratio {:.2f}".format(
                args.target_metric, args.target_sparsity, w1, w2, vote_ratio
            )
        )

    # create model
    if rank == 0:
        print("=> creating model '{}'".format(args.model))
        print("=> M = ", args.M)
    
    model = models.__dict__[args.model](pretrained=True,N = args.N, M = args.M,search = True)
    
    #print(model)

    if rank == 0:
        #print('Pruning with data layout ', args.layout)
        
        print("SGD weight decay = ", args.weight_decay )
        print("SGD momentum =", args.momentum)
        print("SGD learning rate = ", args.finetue_lr)
        print("Weight decay in Sparse Layer = ", args.sparse_decay)
        #print(model.check_N_M())
        print("Vote threshold = ",vote_ratio)
        #print(print_log_sparse_layer(model))
        
    set_decay_sparse_layer(model,args.sparse_decay)
    model.set_datalayout(args.layout)
    if rank == 0:
        print("=> data layout '{}'".format(args.layout))



    global target_sparsity,target_sparsity_erk
    target_sparsity = args.target_sparsity

    target_sparsity_erk = args.target_sparsity


    global erk_sparsity_dict
    erk_sparsity_dict = get_sparsities_erdos_renyi(model.named_layers,target_sparsity_erk,[],list(model.dense_layers),True)

    if rank == 0:
        print("Overall sparsity target of ERK Heuristic ", target_sparsity_erk)
        print("Heuristic ERK sparsity ")
        print(erk_sparsity_dict)


    #exit(0)
    model.to(args.device)
    broadcast_params(model)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.finetue_lr, # change this to fine - tune learning rate 0.0010
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    
    # print(model.parameters())
    model_dir = args.model_dir
    
    start_epoch = 0
    if args.rank == 0 and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    


    if args.rank == 0:
        writer = SummaryWriter(model_dir)
    else:
        writer = None

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])



    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]))


    val_dataset =  datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # Use an explicit sampler even on one GPU to preserve the sample ordering
    # of the original distributed-launch implementation.
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size//args.world_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size//args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer)
        if writer is not None:
            writer.close()
        cleanup_dist()
        return

    niters = len(train_loader)
    
    # does not need adjust lr at this moment
    #lr_scheduler = LRScheduler(optimizer, niters, args)

    best_prec1 = 0.0
    lr_scheduler = LRScheduler(optimizer, niters, args)

    # print("initialize")
    initialize_mod_threshold(model)
    sum_srt,_ = summary(model, input_size=(3, 224, 224), device=args.device)

    set_flops(model)


    flops_layers,params_layers = get_layer_wise_dense_flops_params(model)

    print('Dense Flops of each layer')
    print(flops_layers)

    print('Dense Parameters of each layer')
    print(params_layers)

    # exit(0)
    # print(sum_srt)

    # if rank == 0:
    #     print(sum_srt)

    #     total_sparse_flops,total_dense_flops = compute_flops_reduction(model)
    #     print('Current FLOPs: sparse - {:.4f} M , dense - {:.4f} M, sparse/dense - {:.4f}'.format(
    #         total_sparse_flops*1e-6,total_dense_flops*1e-6, total_sparse_flops/total_dense_flops
    #         )
    #         )





    decay_normalized = normalize_with_flops(model,include_first=True)
    if args.rank ==0:
        print("flops normalization")
        print(decay_normalized)
    


    erk_normal = normalize_erk_sparsity(erk_sparsity_dict)
    
    avg_normalization = average_two_normalization(erk_normal,decay_normalized,w1,w2)
    if args.rank ==0:
        print("use (ERK * {} + flops * {}) as normalization".format(w1,w2))

        print(avg_normalization)

    apply_normalized_factor(model,avg_normalization)
    



    # initialize_layer_sparse_vis_dict(model)

    
    # exit()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, writer)


        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        #do not save
        if rank == 0:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
    if rank == 0:
        print("Best accuracy is ",best_prec1 )
        writer.close()
    cleanup_dist()

def train(train_loader, model, criterion, optimizer,epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # complexity_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    SAD = AverageMeter()

    # switch to train mode
    model.train()
    rank = args.rank

    apply_penalty_flag = False
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        global num_iters


        if num_iters % 5 == 0 :
            # set apply penalty flag as true
            set_apply_penalty_flag(model,True)
            apply_penalty_flag = True
        num_iters  += 1

        # normally the search requires less than 30000 iters
        if num_iters in [30000,40000,50000,60000,70000,80000]:
            if rank == 0:
                print('change args.sparse_decay *=2 at num iterations = {}'.format(num_iters))
            args.sparse_decay *= 2
            set_decay_sparse_layer(model,args.sparse_decay)

        # measure data loading time
        data_time.update(time.time() - end)
        #lr_scheduler.update(i, epoch)
        input_var = input.to(args.device, non_blocking=True)
        target_var = target.to(args.device, non_blocking=True)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        current_lr = get_lr(optimizer)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))

        reduced_loss = reduce_tensor(loss)
        reduced_prec1 = reduce_tensor(prec1)
        reduced_prec5 = reduce_tensor(prec5)

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # if epoch > start_epoch_complexity_loss: 
        #     arch_optimizer.zero_grad()
        loss.backward()
        #loss_complexity.backward()
        
        average_gradients(model)

        if i % 100 == 0:
            #set_debug_flag_sparseConv(model,tmp_flag)
            #adjust_N_M_of_each_layer(model,target_sparsity,epoch)
            adjust_N_M_of_each_layer_based_on_each_group(model,target_sparsity,epoch,i)
            #tmp_flag = ~tmp_flag
        
        optimizer.step()

        if apply_penalty_flag == True:
            set_apply_penalty_flag(model,False)
            apply_penalty_flag = False
            # dsiable penalty 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Complexity_Loss {complexity_losses.val:.4f} ({complexity_losses.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  #'Complexity Learning Rate {current_lr:.3f}\t'
                  'Learning Rate {current_lr:.4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,current_lr=current_lr))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], niter)
            writer.add_scalar('Train/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Train/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Train/Avg_Top5', top5.avg / 100.0, niter)


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    rank = args.rank

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input_var = input.to(args.device, non_blocking=True)
            target_var = target.to(args.device, non_blocking=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target_var, topk=(1, 5))

            reduced_loss = reduce_tensor(loss)
            reduced_prec1 = reduce_tensor(prec1)
            reduced_prec5 = reduce_tensor(prec5)

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(reduced_prec1.item(), input.size(0))
            top5.update(reduced_prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        if rank == 0:
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

            niter = (epoch + 1)
            writer.add_scalar('Eval/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Eval/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Eval/Avg_Top5', top5.avg / 100.0, niter)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_searched_scheme(scheme, metrics=None):
    output_path = args.scheme_output
    if output_path is None:
        output_path = os.path.join(args.model_dir, 'searched_scheme.txt')
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as output_file:
        output_file.write(repr(scheme) + '\n')
    print("Saved searched scheme to '{}'".format(output_path))

    manifest = {
        'schema_version': 1,
        'method': 'domino-mixed-nm',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': {
            'branch': git_value('branch', '--show-current'),
            'commit': git_value('rev-parse', 'HEAD'),
            'dirty': bool(git_value('status', '--porcelain')),
        },
        'objective': {
            'metric': args.target_metric,
            'target_reduction': args.target_sparsity,
            'erk_weight': w1,
            'cost_weight': w2,
            'vote_ratio': vote_ratio,
        },
        'achieved': metrics or {},
        'scheme_file': os.path.abspath(output_path),
        'scheme': scheme,
    }
    manifest_path = output_path + '.json'
    with open(manifest_path, 'w') as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write('\n')
    print("Saved search manifest to '{}'".format(manifest_path))


#================================================================Below are functions for searching===================================================#

# core function
def adjust_N_M_of_each_layer_based_on_each_group(net,target_sparsity,epoch,iterations):

    continuous_sparsity_dict = {}

    change_ = False

    if iterations % 1000 == 0 and args.rank == 0:
        print('Total iterations {}'.format(num_iters))

    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            M = mod.M  
            c_N = mod.N # current c_N
            layer_name = mod.name
            if c_N == 1 : 
                continue
            # elif 'SparseConv0' in layer_name: # you can specify constraints for certain layers if you want
            #     continue
            else:
                if mod.learned_threshold is None:# initialize threshold T for once
                    mod.update_learned_sparsity()
                
                global vote_ratio
                N_intermediate,flag_ , continuous_sparsity,N_inter_change_flag = mod.check_sparsity_each_group(vote_ratio)

                continuous_sparsity_dict[layer_name] = continuous_sparsity
                # layer_sparse_vis_dict[layer_name].append(continuous_sparsity)

                if iterations % 1000 == 0 and args.rank == 0:
                    print('continuous sparsity of layer {} is {:.4f}, current N-inter : M is {} - {}'.format( layer_name,continuous_sparsity,mod.N_intermediate,mod.M) )
                    
            
                if N_inter_change_flag == True: # print if N changes. Not that N_inter may not be in the candidate pool
                    if args.rank == 0:
                        print('change N_inter layer {} to {}'.format(layer_name,mod.N_intermediate))
                    

                if flag_ == True :
                    #print('change {} to N:M {}:{}'.format(layer_name,mod.N,mod.M))
                    current_overall_sparsity = net.get_overall_sparsity() 
                    current_scheme = net.check_N_M()
                    change_ = True
                    current_flops_ratio = check_current_flops_ratio(net)



                    #current_overall_sparsity_N_inter_M = get_overall_sparsity_N_inter_M(net)

                    if args.rank == 0:
                        print('Decision made at iterations {}: change scheme of layer {} to {} - {}'.format(
                        num_iters,mod.name,mod.N,mod.M))
                        print('Current sparse flops ratio is {}'.format(current_flops_ratio))
                        print('Current overall sparsity is {}'.format(current_overall_sparsity))
                        print('Schemes at iterations {} is '.format(num_iters))
                        print(net.check_N_M())

                    flops_normalization = normalize_with_flops(net,include_first=True)

                    global target_sparsity_erk,erk_sparsity_dict

                    erk_sparsity = update_erk_sparsity(erk_sparsity_dict,net)
                    erk_normal = normalize_erk_sparsity(erk_sparsity)

                    global w1,w2
                    
                    new_normalized = average_two_normalization(erk_normal,flops_normalization,w1,w2)

                    # new_normalized = flops_normalization

                    if args.rank == 0:
                        print('new flops normalization')
                        print(flops_normalization)
                        print('new erk normalization')
                        print(erk_normal)
                        print("updateing normalizaion with (erk * {} + flops * {})".format(w1,w2))
                        print(new_normalized)
                        print('\n')


                    apply_normalized_factor(net,new_normalized)

                    decision_dict[current_overall_sparsity] = (iterations,current_scheme)
                    
                    
                    current_target_value = (
                        current_overall_sparsity
                        if args.target_metric == 'params'
                        else current_flops_ratio
                    )
                    if current_target_value >= target_sparsity-0.001:
                        if args.rank == 0:
                            print(
                                'Target {} reduction {:.5f} has been achieved; current value is {:.5f}'.format(
                                    args.target_metric, target_sparsity, current_target_value
                                )
                            )
                            print('The schemes of each layer')
                            print(net.check_N_M())
                            total_sparse_flops,total_dense_flops = compute_flops_reduction(net)
                            print('Current FLOPs: sparse - {:.4f} M , dense - {:.4f} M, sparse/dense - {:.4f}'.format(
                                total_sparse_flops*1e-6,total_dense_flops*1e-6, total_sparse_flops/total_dense_flops
                                )
                                )
                            # print('Decision logs')
                            # print(decision_dict)
                            save_searched_scheme(
                                net.check_N_M(),
                                {
                                    'parameter_reduction': current_overall_sparsity,
                                    'flops_reduction': current_flops_ratio,
                                    'iterations': num_iters,
                                    'epoch': epoch,
                                },
                            )
                        exit(0)



# def check_contributions_gradient_and

def average_two_normalization(norm_dict1,norm_dict2,w1=0.5,w2=0.5):
    if norm_dict1 is None:
        return norm_dict2
    elif norm_dict2 is None:
        return norm_dict1

    new_dict = {}
    for layer_name, value in norm_dict1.items():
        new_dict[layer_name] = (value*w1 + norm_dict2[layer_name]*w2)
    
    return new_dict



def set_apply_penalty_flag(net,flag):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            mod.apply_penalty = flag

def set_debug_flag_sparseConv(net,flag):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            mod.print_flag = flag


def initialize_mod_threshold(net):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            if mod.learned_threshold is None:
                mod.update_learned_sparsity()





    


def set_flops(net):
    for mod in net.modules():
        if isinstance(mod, SparseConv): 
            # dense parameters
            in_shape = np.array(mod.input_shape)
            
            out_shape = np.array(mod.output_shape)
            mod.flops =  conv_flops_counter_hook(mod,in_shape,out_shape)

            print('{} input-{} output-{} FLOPs-{}M'.format(mod.name,in_shape,out_shape,mod.flops*1e-6))
        elif isinstance(mod, SparseLinear): 
            input_ = mod.in_features
            # pytorch checks dimensions, so here we don't care much
            output_last_dim = mod.out_features
            bias_flops = output_last_dim if mod.bias is not None else 0

            mod.flops = int(input_ * output_last_dim + bias_flops)
            print('{} input-{} output-{} FLOPs-{}M'.format(mod.name,input_,output_last_dim,mod.flops*1e-6))



# dense/sparse
def check_current_flops_ratio(net):
    sparse_flops = 0
    dense_flops = 0

    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 

            dense_flops += mod.flops
            sparse_flops += mod.flops * mod.N/mod.M   

    
    return 1.0  - (sparse_flops/dense_flops)


def apply_normalized_factor(net,decay_normalized):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            layer_name = mod.name
            mod.normalized_factor = decay_normalized[layer_name]



def print_log_sparse_layer(net):
    log_dict = {}
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            layer_name = mod.name
            log_dict[layer_name] = mod.log
    
    return log_dict



def get_N_inter_M(net):
    sparse_scheme = {}
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            layer_name = mod.name
            sparse_scheme[mod.get_name()] = list([mod.N_intermediate,mod.M])
    return sparse_scheme



def get_overall_sparsity_N_inter_M(net):
    sparse_paras = 0
    dense_paras = 0
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            sparse_paras += mod.dense_parameters * mod.N_intermediate / mod.M 
            dense_paras += mod.dense_parameters

    return 1.0 - (sparse_paras/dense_paras)  


def normalize_with_flops(net,include_first=True):
    decay_normalized = {}

# and include_first==True 
    normalized_factor = []
    layer_list = []
    valid_flops_list = []
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) : 
            layer_name = mod.name
            layer_list.append(layer_name)
            dense_flops = mod.flops
            sparse_flops = mod.flops * mod.N / mod.M  
            normalized_factor.append(sparse_flops)
            if 'SparseConv0' in layer_name:
                if include_first == True:
                    valid_flops_list.append(sparse_flops)
                else:
                    continue
            else:        
                valid_flops_list.append(sparse_flops)

        

    valid_flops_list = np.asarray(valid_flops_list)

    normalized_factor = np.asarray(normalized_factor)
    normalized_factor = normalized_factor/np.amax(valid_flops_list)

    normalized_factor = normalized_factor.tolist()
    zip_iterator = zip(layer_list, normalized_factor)
    decay_normalized = dict(zip_iterator)

    return decay_normalized




# return total sparse and dense flops
def compute_flops_reduction(net):
    total_dense_flops = 0
    total_sparse_flops = 0
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) : 
            layer_name = mod.name
            
            dense_flops = mod.flops
            sparse_flops = mod.flops * mod.N / mod.M  
            total_dense_flops += dense_flops
            total_sparse_flops += sparse_flops
    
    return total_sparse_flops,total_dense_flops




            
def set_decay_sparse_layer(net,decay):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            mod.decay = decay



# def check_sparse_pattern(network,N,M):
def adjust_learning_rate(optimizer, epoch):
    #global state
    if epoch in args.schedule:
        args.finetue_lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.finetue_lr

def adjust_learning_rate_complexity(optimizer, epoch):
    #global state
    if epoch in args.schedule:
        args.lra *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lra

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    main()
