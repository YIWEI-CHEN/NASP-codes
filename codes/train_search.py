import math
from copy import deepcopy

import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import logging.handlers
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils import data
from tensorboard_logger import configure, log_value

import utils
from model_search import Network
from architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--train_batch_size', type=int, default=68, help='train batch size')
parser.add_argument('--valid_batch_size', type=int, default=72, help='valid batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default="0,1", help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-5, help='weight decay for arch encoding')
parser.add_argument('--name', type=str, default="runs", help='name for log')
parser.add_argument('--debug', action='store_true', default=False, help='debug or not')
parser.add_argument('--greedy', type=float, default=0, help='explore and exploitation')
parser.add_argument('--l2', type=float, default=0, help='additional l2 regularization for alphas')
parser.add_argument('--exec_script', type=str, default='scripts/dist_search.sh', help='script to run exp')
parser.add_argument('--dist-url', default='tcp://datalab.cse.tamu.edu:50017', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--subproblem_batch_ratio', type=float, default=0.25,
                    help='the portion of samples used for subproblem')
parser.add_argument('--eta', type=float, default=1.0, help="the weight for average gradient")
parser.add_argument('--mu', type=float, default=0.1, help="the weight for regularized difference")
parser.add_argument('--subproblem_maximum_iterations', type=int, default=5, help='num of subproblem_maximum_iterations')
CIFAR_CLASSES = 10


def main():
  try:
    log_queue = None
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'), exec_script=args.exec_script)

    # Logging configuration
    utils.setup_logger(args)
    root = logging.getLogger()

    if not torch.cuda.is_available():
      root.info('no gpu device available')
      sys.exit(1)

    # Log thread to receive log from child processes
    log_thread, log_queue = utils.run_log_thread()

    root.info('gpu device = {}'.format(args.gpu))
    root.info("args = %s", args)

    ngpus_per_node = torch.cuda.device_count()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, log_queue))
  finally:
    # close the thread
    if log_queue is not None:
      log_queue.put(None)
      log_thread.join(1.0)


def main_worker(gpu, ngpus_per_node, args, log_queue):
  qh = logging.handlers.QueueHandler(log_queue)
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  root.addHandler(qh)

  args.gpu = gpu

  # Fix seed
  utils.fix_seed(args.seed)

  try:
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    # tensorboard_logger configuration
    configure('{}/{}_{}'.format(args.save, args.name, args.rank))

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # create model
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.greedy, args.l2, gpu=args.gpu)
    root.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # define optimizer
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.train_batch_size = int(args.train_batch_size / ngpus_per_node)
    args.valid_batch_size = int(args.valid_batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # Meta architecture
    architect = Architect(model, args)

    # Data loading code
    train_queue, valid_queue, all_valid_queue = get_train_validation_loader(args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    best_acc = 0
    for epoch in range(args.epochs):
      lr = scheduler.get_lr()[0]
      log_value("lr", lr, epoch)
      root.info('epoch %d lr %e', epoch, lr)

      genotype = model.genotype()
      root.info('genotype = %s', genotype)

      # training
      start_time = time.time()
      train_acc, train_obj, alphas_time, forward_time, backward_time = \
        train(train_queue, valid_queue, model, optimizer, args, epoch)
      end_time = time.time()
      root.info('train time: {}'.format(end_time - start_time))
      root.info("alpha_time: {}".format(alphas_time))
      root.info("forward_time: {}".format(forward_time))
      root.info("backward_time: {}".format(backward_time))
      log_value('train_acc', train_acc, epoch)
      root.info("train_acc: {}".format(train_acc))

      # validation
      if args.rank == 0:
        start_time2 = time.time()
        valid_acc, valid_obj = infer(all_valid_queue, model, criterion, args)
        end_time2 = time.time()
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)

        root.info("inference time %f", end_time2 - start_time2)
        log_value('valid_acc', valid_acc, epoch)
        root.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        if is_best:
          root.info('Epoch {} produces best valid_acc {}, best arch:\n{}'.format(
            epoch, best_acc, genotype
          ))
        # root.info('alphas_normal = %s', model.alphas_normal)
        # root.info('alphas_reduce = %s', model.alphas_reduce)

      # update learning rate
      scheduler.step()


  except Exception as e:
    root.error(e)
    dist.destroy_process_group()


def train(train_queue, valid_queue, model, optimizer, args, epoch):
  root = logging.getLogger()
  rank = args.rank
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  alphas_time = 0
  forward_time = 0
  backward_time = 0
  arch_dimension = sum([a.numel() for a in model.arch_parameters()])
  weight_dimension = sum([w.numel() for w in model.parameters()])
  total_batchs = len(train_queue)

  model.train()
  for step, (inputs, targets) in enumerate(train_queue):
    begin1 = time.time()
    # fix weights and update arch
    # architect.step(input, target, input_search, target_search, lr, optimizer)
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda(args.gpu, non_blocking=True)
    target_search = target_search.cuda(args.gpu, non_blocking=True)

    # compute local arch grad
    local_arch_grad = get_arch_gradient(model, input_search, target_search, arch_weight_decay=args.arch_weight_decay)
    # send to rank 0
    # root.info('before reducing local_arch_grad: {}'.format(local_arch_grad[0:5]))
    dist.reduce(local_arch_grad, 0)

    # average local_arch_grad
    if rank == 0:
      arch_grad = local_arch_grad * 1.0 / args.world_size
    else:
      arch_grad = torch.zeros(arch_dimension, 1, device=args.gpu)

    # sync arch_grad from rank 0
    dist.broadcast(arch_grad, 0)
    # root.info('after broadcasting arch_grad: {}'.format(arch_grad[0:5]))

    # compute local solution of subproblem of InexactDANE
    local_subproblem_solution = get_InexactDANE_subproblem_solution(model, arch_grad, input_search, target_search, args)
    # send to rank 0
    # root.info('before reducing local_subproblem_solution: {}'.format(local_subproblem_solution[0:5]))
    dist.reduce(local_subproblem_solution, 0)

    # average local_subproblem_solution
    if rank == 0:
      new_arch_parameters = local_subproblem_solution * 1.0 / args.world_size
    else:
      new_arch_parameters = torch.zeros(arch_dimension, 1, device=args.gpu)

    # sync new_arch_parameters from rank 0
    dist.broadcast(new_arch_parameters, 0)
    # root.info('after broadcasting new_arch_parameters: {}'.format(new_arch_parameters[0:5]))

    # update the architecture parameters
    # root.info('Updating arch parameters')
    model.update_arch(new_arch_parameters, replace=True)
    end1 = time.time()
    alphas_time += end1 - begin1

    if args.rank == 0 and step % args.report_freq == 0:
      _step = epoch * total_batchs + step
      log_arch('normal', model.alphas_normal, _step)
      log_arch('reduce', model.alphas_reduce, _step)

    # fix arch and update weights
    n = inputs.size(0)
    inputs = inputs.cuda(args.gpu, non_blocking=True)
    targets = targets.cuda(args.gpu, non_blocking=True)
    optimizer.zero_grad()

    # compute local weight grad
    local_weight_grad, loss, prec1, prec5, _forward_time, _backward_time \
      = get_weight_gradient(model, inputs, targets)
    forward_time += _forward_time
    backward_time += _backward_time
    # send to rank 0
    # root.info('before reducing local_weight_grad: {}'.format(local_weight_grad[0:5]))
    dist.reduce(local_weight_grad, 0)

    # average local_weight_grad
    if rank == 0:
      weight_grad = local_weight_grad * 1.0 / args.world_size
    else:
      weight_grad = torch.zeros(weight_dimension, 1, device=args.gpu)

    # sync weight_grad from rank 0
    dist.broadcast(weight_grad, 0)
    # root.info('after broadcasting weight_grad: {}'.format(weight_grad[0:5]))

    # weight_grad = local_weight_grad  # should comment out
    model.update_weight_grad(weight_grad)
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    # update weights
    optimizer.step()
    model.clip()

    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      root.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg, alphas_time, forward_time, backward_time


def infer(valid_queue, model, criterion, args):
  root = logging.getLogger()

  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  model.binarization()
  for step, (inputs, targets) in enumerate(valid_queue):
    inputs = inputs.cuda(args.gpu, non_blocking=True)
    targets = targets.cuda(args.gpu, non_blocking=True)

    logits = model(inputs)
    loss = criterion(logits, targets)

    prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
    n = inputs.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      root.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  model.restore()
  return top1.avg, objs.avg


def get_train_validation_loader(args):
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  split = int(np.floor(args.train_portion * num_train))
  num_valid = num_train - split

  local_num_train = int(math.ceil(1.0 * split / args.world_size))
  local_train_set = data.Subset(train_data,
                                range(local_num_train * args.rank,
                                      local_num_train * (args.rank + 1)))

  local_num_valid = int(math.ceil(1.0 * num_valid / args.world_size))
  local_valid_set = data.Subset(train_data,
                                range(split + local_num_valid * args.rank,
                                      split + local_num_valid * (args.rank + 1)))
  train_queue = torch.utils.data.DataLoader(
    local_train_set, batch_size=args.train_batch_size,
    num_workers=args.workers, pin_memory=True, shuffle=True)
  train_queue.size = local_num_train

  valid_queue = torch.utils.data.DataLoader(
    local_valid_set, batch_size=args.valid_batch_size,
    num_workers=args.workers, pin_memory=True, shuffle=True)
  valid_queue.name = 'local_valid'
  valid_queue.size = local_num_valid

  if args.rank == 0:
    all_valid_set = data.Subset(train_data,
                                range(split, num_train))
    all_valid_queue = torch.utils.data.DataLoader(
      all_valid_set, batch_size=args.valid_batch_size * args.world_size,
      num_workers=args.workers, pin_memory=True, shuffle=True)
    all_valid_queue.size = split
    all_valid_queue.name = 'all_valid'
  else:
    all_valid_queue = None

  return train_queue, valid_queue, all_valid_queue


def _concat(xs):
  return torch.cat([x.reshape(-1, 1) for x in xs], dim=0)


def get_weight_gradient(model, inputs, targets):
  """
  After reading SGD code in pytorch, I will not
    1. Add weight decay to gradient
  """
  # binarize arch parameters
  model.binarization()

  # forward, compute loss
  begin = time.time()
  logits = model(inputs, updateType="weights")
  loss = model._criterion(logits, targets)
  forward_time = time.time() - begin

  # backward, get gradient
  model.zero_grad()
  begin = time.time()
  loss.backward()
  backward_time = time.time() - begin

  # flat gradient to a n-dim vector
  flat_gradient = torch.cat(
    [v.grad.reshape(-1, 1) if v.grad is not None else torch.zeros_like(v).reshape(-1, 1) for v in  model.parameters()],
    dim=0).data * 1.0 / inputs.size(0)

  # restore arch parameters
  model.restore()

  # performance
  prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))

  return flat_gradient, loss, prec1, prec5, forward_time, backward_time


def get_InexactDANE_subproblem_solution(model, arch_grad, inputs, targets, args):
  """Returns a solution to the local InexactDANE subproblem using SVRG + Adam."""
  beta1 = 0.9
  beta2 = 0.999
  eps = 1e-8
  subproblem_maximum_iterations = args.subproblem_maximum_iterations
  sub_batch_size = int(inputs.size(0) * args.subproblem_batch_ratio)
  total_inputs = inputs.size(0)
  sample_indices = np.random.choice(total_inputs, subproblem_maximum_iterations)

  arch_parameters = _concat(model.arch_parameters()).data
  new_arch_parameters = deepcopy(arch_parameters)
  exp_avg = torch.zeros_like(arch_parameters)
  exp_avg_sq = torch.zeros_like(arch_parameters)

  for step, idx in enumerate(sample_indices):
    end = idx + sub_batch_size
    sample_inputs, sample_targets = inputs[idx:end], targets[idx:end]
    # gradient of current arch
    sample_arch_grad = get_arch_gradient(
      model, sample_inputs, sample_targets, args.arch_weight_decay, arch_parameters)
    # gradient of updated arch
    sample_new_arch_grad = get_arch_gradient(
      model, sample_inputs, sample_targets, args.arch_weight_decay, new_arch_parameters)
    # compute updated direction
    update_direction = sample_new_arch_grad - sample_arch_grad + \
                       args.eta * arch_grad + args.mu * (new_arch_parameters - arch_parameters)
    # Use Adam to update new arch
    exp_avg.mul_(beta1).add_(1 - beta1, update_direction)
    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, update_direction, update_direction)
    denom = exp_avg_sq.sqrt().add_(eps)

    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)
    step_size = args.arch_learning_rate * math.sqrt(bias_correction2) / bias_correction1
    new_arch_parameters.addcdiv_(-step_size, exp_avg, denom)

  return new_arch_parameters


def get_arch_gradient(model, input_, target_, arch_weight_decay=0.0, arch_parameter=None):
  # set arch gradient zero
  model.zero_arch_grad()

  # use specified arch values
  if arch_parameter is not None:
    model.update_arch(arch_parameter, replace=False)

  # binarize arch parameters
  model.binarization()

  # loss
  loss = model._loss(input_, target_, updateType="alphas")

  # arch gradient
  grad = torch.autograd.grad(loss, model.arch_parameters())

  # restore from binarization
  model.restore(usage='binary')

  for i, arch in enumerate(model.arch_parameters()):
    # weight decay
    if arch_weight_decay != 0:
      arch.grad = grad[i] * 1.0 / input_.size(0) + arch_weight_decay * arch.data
    else:
      arch.grad = grad[i] * 1.0 / input_.size(0)

  # flat grad
  flat_grad = _concat([arch.grad for arch in model.arch_parameters()]).data

  # restore from svrg usage
  if arch_parameter is not None:
    model.restore(usage='svrg')

  return flat_grad


def log_arch(cell, arch, step):
  root = logging.getLogger()
  root.info('{} param min {:.4f}, max {:.4f}, mean {:.4f}, std {:.4f}'.format(
    cell, arch.min(), arch.max(), arch.mean(), arch.std()))
  root.info('{} param: {}'.format(cell, arch))
  log_value('{}_min'.format(cell), arch.min(), step)
  log_value('{}_max'.format(cell), arch.max(), step)
  log_value('{}_mean'.format(cell), arch.mean(), step)
  log_value('{}_std'.format(cell), arch.std(), step)

  root.info('{} grad min {:.4f}, max {:.4f}, mean {:.4f}, std {:.4f}'.format(
    cell, arch.grad.min(), arch.grad.max(), arch.grad.mean(), arch.grad.std()))
  # root.info('{} grad: {}'.format(cell, arch.grad))
  log_value('{}_grad_min'.format(cell), arch.grad.min(), step)
  log_value('{}_grad_max'.format(cell), arch.grad.max(), step)
  log_value('{}_grad_mean'.format(cell), arch.grad.mean(), step)
  log_value('{}_grad_std'.format(cell), arch.grad.std(), step)


if __name__ == '__main__':
  main()

