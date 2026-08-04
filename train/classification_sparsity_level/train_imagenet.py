from __future__ import division
import argparse
import os
import os.path as osp
import sys
import time

# Make train/devkit importable regardless of the working directory used to
# launch this script.
TRAIN_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if TRAIN_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_ROOT)
REPO_ROOT = osp.abspath(osp.join(TRAIN_ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import yaml
from tensorboardX import SummaryWriter
import models
from devkit.sparse_ops import SparseConv,SparseLinear
from devkit.core import (
    LRScheduler,
    average_gradients,
    broadcast_params,
    cleanup_dist,
    get_device,
    init_dist,
    apply_parameter_masks,
    load_initial_checkpoint,
    load_parameter_masks,
    load_state,
    load_state_ckpt,
    reduce_tensor,
    register_parameter_mask_hooks,
    save_checkpoint,
    set_sparse_scheme,
)

from devkit.core import load_state_file
from devkit.dataset.imagenet_dataset import ColorAugmentation, ImagenetDataset
from imagenet_data import ParquetImageNetDataset
from pruning.unstructured_magnitude.gradual import (
    GradualMagnitudeConfig,
    apply_gradual_pruning,
    eligible_weight_parameters,
    initialize_masks,
    mask_statistics,
    restore_masks_from_training_checkpoint,
    save_mask_artifact,
    scheduled_sparsity,
    should_update_masks,
)



# Sparse
import ast # for read schemes from txt file
parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='configs/config_resnet50_2:4.yaml')
parser.add_argument('--schemes_file', default='schemes/test.txt')
parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int)
parser.add_argument(
    '--port', default=29500, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--label_smoothing', default=0.0,type=float)
parser.add_argument('--momentum', default=0.9,type=float)
parser.add_argument('--base_lr', default=0.1,type=float)
parser.add_argument('--weight_decay', default=0.00005,type=float)
parser.add_argument(
    '--decay',
    default=0.002,
    type=float,
    help='sparse weight-penalty coefficient (the original ResNet configs use 0.002)',
)
parser.add_argument('--model_dir', type=str,  default='resnet56_cifar/resnet56_M')
parser.add_argument('--resume_from', default='', help='resume_from')
parser.add_argument(
    '--initial-checkpoint',
    default='',
    help='exact checkpoint used to initialize a new fine-tuning run',
)
parser.add_argument(
    '--weight-mask-file',
    default='',
    help='parameter masks kept fixed during fine-tuning',
)
parser.add_argument(
    '--dataset-format',
    choices=('meta', 'imagefolder', 'parquet'),
    default='meta',
    help='meta preserves the original loader; parquet streams local Drive shards',
)
parser.add_argument('--data-root', default='', help='ImageFolder root containing train/val')
parser.add_argument('--parquet-root', default='', help='root containing ImageNet Parquet shards')
parser.add_argument('--train-parquet-pattern', default='data/train-*.parquet')
parser.add_argument('--val-parquet-pattern', default='data/validation-*.parquet')
parser.add_argument('--train-num-samples', type=int, default=1281167)
parser.add_argument('--val-num-samples', type=int, default=50000)
parser.add_argument('--shuffle-buffer', type=int, default=10000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument(
    '--save-every-epoch',
    action='store_true',
    help=(
        'write a resumable checkpoint after every epoch; disabled by default '
        'to preserve the original epoch > 1 save schedule'
    ),
)
parser.add_argument(
    '--data-workers',
    type=int,
    default=None,
    help='override YAML workers without changing the original config',
)
parser.add_argument(
    '--gradual-pruning-target',
    type=float,
    default=None,
    help='enable gradual magnitude pruning and set final eligible-weight sparsity',
)
parser.add_argument('--gradual-pruning-start-epoch', type=int, default=0)
parser.add_argument('--gradual-pruning-end-epoch', type=int, default=None)
parser.add_argument('--gradual-pruning-frequency', type=int, default=1)
parser.add_argument('--gradual-pruning-power', type=float, default=3.0)
parser.add_argument(
    '--gradual-pruning-scope',
    choices=('global', 'local'),
    default='global',
)
parser.add_argument('--gradual-prune-first', action='store_true')
parser.add_argument('--gradual-prune-last', action='store_true')
parser.add_argument(
    '--gradual-mask-dir',
    default='',
    help='directory for immutable epoch-specific gradual mask artifacts',
)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

args = None






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
    if args.decay < 0.0:
        raise ValueError('--decay cannot be negative')
    if args.data_workers is not None:
        if args.data_workers < 0:
            raise ValueError('--data-workers cannot be negative')
        args.workers = args.data_workers

    args.gradual_pruning_config = None
    if args.gradual_pruning_target is not None:
        if args.weight_mask_file:
            raise ValueError(
                '--weight-mask-file cannot be combined with gradual pruning; '
                'resume gradual runs through --model_dir instead'
            )
        gradual_end_epoch = (
            args.epochs - 1
            if args.gradual_pruning_end_epoch is None
            else args.gradual_pruning_end_epoch
        )
        args.gradual_pruning_config = GradualMagnitudeConfig(
            target_sparsity=args.gradual_pruning_target,
            start_epoch=args.gradual_pruning_start_epoch,
            end_epoch=gradual_end_epoch,
            frequency=args.gradual_pruning_frequency,
            power=args.gradual_pruning_power,
            scope=args.gradual_pruning_scope,
            prune_first=args.gradual_prune_first,
            prune_last=args.gradual_prune_last,
        )
        args.gradual_pruning_config.validate(args.epochs)

    rank, world_size = init_dist(
        backend='nccl', port=args.port )
    args.rank = rank
    args.world_size = world_size
    args.device = get_device()
    if rank == 0:
        mode = "distributed ({} GPUs)".format(world_size) if world_size > 1 else "single GPU"
        print("Running in {} mode on {}.".format(mode, args.device))

    # create model
    decay = args.decay
    if rank == 0:
        print("=> creating model '{}'".format(args.model))


    # if args.resume_from=='':
    #     model = models.__dict__[args.model](pretrained=True,N = args.N, M = args.M) #NHWC
    # else:
    #     model = models.__dict__[args.model](pretrained=False,N = args.N, M = args.M) #NHWC
    #     load_state_file(args.resume_from,model)

    model = models.__dict__[args.model](
        pretrained=not bool(args.initial_checkpoint), N=args.N, M=args.M
    )
    if args.initial_checkpoint:
        load_initial_checkpoint(args.initial_checkpoint, model)
        if rank == 0:
            print("Loaded exact initial checkpoint '{}'".format(args.initial_checkpoint))

    #model.set_datalayout('NHWC')

    if args.gradual_pruning_config is not None:
        # Gradual unstructured pruning must not be mixed with N:M. Force every
        # sparse operator to dense N:M while the magnitude masks evolve.
        sparse_schemes = {
            layer.get_name(): [args.M, args.M]
            for layer in model.modules()
            if isinstance(layer, (SparseConv, SparseLinear))
        }
        if not sparse_schemes:
            raise ValueError('Model contains no sparse layers for a dense scheme')
    else:
        with open(args.schemes_file) as f:
            first_line = f.readline()

        if not first_line.strip():
            raise ValueError("The schemes file is empty: {}".format(args.schemes_file))

        sparse_schemes = ast.literal_eval(first_line)


    # set layer-wise sparse scheme    
    set_sparse_scheme(model,sparse_schemes)

    set_weight_penalty(model,decay)


    # summary(model, input_size=(3, 224, 224))

    # set_flops(model)

    if rank == 0:
        if args.gradual_pruning_config is not None:
            print('Use internally generated dense N:M scheme for gradual pruning')
        else:
            print('Use schemes file {}'.format(args.schemes_file))
        print("Start to train mixed Sparse NN")
        print(model)
        # print(model.named_layers)
        #print(model.dense_layers)
        print("Sparse Scheme")
        print(model.check_N_M())


    model.to(args.device)
    broadcast_params(model)

        
    #print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # auto resume from a checkpoint
    model_dir = args.model_dir
    start_epoch = 0
    if args.rank == 0 and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.evaluate:
        load_state_ckpt(args.checkpoint_path, model)
    else:
        best_prec1, start_epoch = load_state(model_dir, model, optimizer=optimizer)

    args.parameter_masks = {}
    args.parameter_mask_hooks = []
    args.gradual_eligible = []
    args.gradual_protected = []
    if args.weight_mask_file:
        args.parameter_masks = load_parameter_masks(
            args.weight_mask_file, model, args.device
        )
        apply_parameter_masks(model, args.parameter_masks)
        args.parameter_mask_hooks = register_parameter_mask_hooks(
            model, args.parameter_masks
        )
        if rank == 0:
            print(
                "Loaded {} persistent parameter mask(s) from '{}'".format(
                    len(args.parameter_masks), args.weight_mask_file
                )
            )
    elif args.gradual_pruning_config is not None:
        args.gradual_eligible, args.gradual_protected = eligible_weight_parameters(
            model,
            prune_first=args.gradual_pruning_config.prune_first,
            prune_last=args.gradual_pruning_config.prune_last,
        )
        args.parameter_masks = restore_masks_from_training_checkpoint(
            model_dir,
            args.gradual_eligible,
            args.gradual_pruning_config,
        )
        if args.parameter_masks is None:
            args.parameter_masks = initialize_masks(
                args.gradual_eligible,
                preserve_existing_zeros=start_epoch > 0,
            )
        apply_parameter_masks(model, args.parameter_masks)
        args.parameter_mask_hooks = register_parameter_mask_hooks(
            model, args.parameter_masks
        )
        args.gradual_mask_dir = args.gradual_mask_dir or osp.join(
            model_dir, 'gradual-masks'
        )
        if rank == 0:
            restored = mask_statistics(
                args.gradual_eligible, args.parameter_masks
            )['eligible_sparsity']
            print(
                'Enabled gradual {} magnitude pruning to {:.2f}% from epoch {} '
                'through {} (power {:.2f}); restored mask sparsity {:.2f}%.'.format(
                    args.gradual_pruning_config.scope,
                    100.0 * args.gradual_pruning_config.target_sparsity,
                    args.gradual_pruning_config.start_epoch,
                    args.gradual_pruning_config.end_epoch,
                    args.gradual_pruning_config.power,
                    100.0 * restored,
                )
            )
    if args.rank == 0:
        writer = SummaryWriter(model_dir)
    else:
        writer = None

    cudnn.benchmark = True





    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorAugmentation(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if args.dataset_format == 'meta':
        train_dataset = ImagenetDataset(
            args.train_root, args.train_source, train_transform
        )
        val_dataset = ImagenetDataset(args.val_root, args.val_source, val_transform)
    elif args.dataset_format == 'imagefolder':
        if not args.data_root:
            raise ValueError('--data-root is required for imagefolder datasets')
        train_dataset = datasets.ImageFolder(
            osp.join(args.data_root, 'train'), transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            osp.join(args.data_root, 'val'), transform=val_transform
        )
    else:
        if not args.parquet_root:
            raise ValueError('--parquet-root is required for parquet datasets')
        train_dataset = ParquetImageNetDataset(
            args.parquet_root,
            args.train_parquet_pattern,
            'train',
            train_transform,
            args.train_num_samples,
            shuffle=True,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
            rank=rank,
            world_size=world_size,
        )
        val_dataset = ParquetImageNetDataset(
            args.parquet_root,
            args.val_parquet_pattern,
            'validation',
            val_transform,
            args.val_num_samples,
            shuffle=False,
            seed=args.seed,
            shuffle_buffer=1,
            rank=rank,
            world_size=world_size,
        )


    # Use an explicit sampler even on one GPU to preserve the sample ordering
    # of the original distributed-launch implementation.
    if args.dataset_format == 'parquet':
        train_sampler = None
        val_sampler = None
    else:
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
        shuffle=False, num_workers=args.workers, pin_memory=args.device.type == 'cuda',
        sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size//args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.type == 'cuda', sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer)
        if writer is not None:
            writer.close()
        cleanup_dist()
        return

    niters = len(train_loader)

    lr_scheduler = LRScheduler(optimizer, niters, args)



    


    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        else:
            train_dataset.set_epoch(epoch)

        if (
            args.gradual_pruning_config is not None
            and should_update_masks(args.gradual_pruning_config, epoch)
        ):
            target = scheduled_sparsity(args.gradual_pruning_config, epoch)
            statistics = apply_gradual_pruning(
                args.gradual_eligible,
                args.parameter_masks,
                target,
                args.gradual_pruning_config.scope,
            )
            if rank == 0:
                mask_path = osp.join(
                    args.gradual_mask_dir,
                    'gradual-mask-epoch-{}.pth'.format(epoch + 1),
                )
                save_mask_artifact(
                    mask_path,
                    args.parameter_masks,
                    args.gradual_pruning_config,
                    epoch,
                    target,
                    statistics,
                    args.gradual_protected,
                )
                print(
                    'Gradual pruning epoch {}: scheduled {:.2f}%, actual {:.2f}%, '
                    'mask {}'.format(
                        epoch + 1,
                        100.0 * target,
                        100.0 * statistics['eligible_sparsity'],
                        mask_path,
                    )
                )

        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, writer)


        prec1 = validate(val_loader, model, criterion, epoch, writer)

        if rank == 0:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if args.save_every_epoch or epoch > 1:
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                    #'arch_optimizer': arch_optimizer.state_dict(),
                }
                if args.gradual_pruning_config is not None:
                    checkpoint_state['parameter_masks'] = {
                        name: mask.detach().to(device='cpu', dtype=torch.bool)
                        for name, mask in args.parameter_masks.items()
                    }
                    checkpoint_state['gradual_pruning'] = {
                        'schedule': args.gradual_pruning_config.to_dict(),
                        'epoch': epoch,
                        'scheduled_target': scheduled_sparsity(
                            args.gradual_pruning_config, epoch
                        ),
                        'statistics': mask_statistics(
                            args.gradual_eligible, args.parameter_masks
                        ),
                        'protected_parameters': args.gradual_protected,
                    }
                save_checkpoint(model_dir, checkpoint_state, is_best)
    if rank == 0:
        print("Best accuracy is ",best_prec1 )
        writer.close()
    cleanup_dist()

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #complexity_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    SAD = AverageMeter()

    # switch to train mode
    model.train()
    rank = args.rank

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.update(i, epoch)

        input_var = input.to(args.device, non_blocking=True)
        target_var = target.to(args.device, non_blocking=True)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        current_lr = lr_scheduler.get_lr()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))

        reduced_loss = reduce_tensor(loss)
        reduced_prec1 = reduce_tensor(prec1)
        reduced_prec5 = reduce_tensor(prec5)

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        
        optimizer.zero_grad()
        
        loss.backward()
        
        average_gradients(model)
        optimizer.step()
        if args.parameter_masks:
            apply_parameter_masks(model, args.parameter_masks)
       

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
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




def set_weight_penalty(model, decay):
    for mod in model.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) : 
            mod.decay = decay



if __name__ == '__main__':
    main()
