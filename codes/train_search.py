import os
import sys
import time
import glob
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from tensorboard_logger import configure, log_value
import pdb


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
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
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--name', type=str, default="runs", help='name for log')
parser.add_argument('--debug', action='store_true', default=False, help='debug or not')
parser.add_argument('--greedy', type=float, default=0, help='explore and exploitation')
parser.add_argument('--l2', type=float, default=0, help='additional l2 regularization for alphas')
parser.add_argument('--exec_script', type=str, default='scripts/search.sh', help='script to run exp')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
args = parser.parse_args()

args.train_batch_size = args.batch_size
args.valid_batch_size = args.batch_size

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if args.debug:
  args.save += "_debug"
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'), exec_script=args.exec_script)

# Logging configuration
utils.setup_logger(args)

# tensorboard_logger configuration
configure(args.save + "/%s"%(args.name))

CIFAR_CLASSES = 10

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


def main():
  root = logging.getLogger()

  if not torch.cuda.is_available():
    root.info('no gpu device available')
    sys.exit(1)

  # Fix seed
  utils.fix_seed(args.seed)

  root.info('gpu device = %d' % args.gpu)
  root.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.greedy, args.l2)
  model = model.cuda()
  root.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # Data loading code
  train_queue, train_sampler, valid_queue = utils.get_train_validation_loader(args)
  test_queue = utils.get_test_loader(args)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  best_acc = 0
  for epoch in range(args.epochs):
    lr = scheduler.get_lr()[0]
    log_value("lr", lr, epoch)
    root.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    root.info('genotype = %s', genotype)

    # training
    architect.alpha_forward = 0
    architect.alpha_backward = 0
    start_time = time.time()
    train_acc, train_obj, alphas_time, forward_time, backward_time = \
      train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
    end_time = time.time()
    root.info("train time %f", end_time - start_time)
    root.info("alphas_time %f ", alphas_time)
    root.info("forward_time %f", forward_time)
    root.info("backward_time %f", backward_time)
    root.info("alpha_forward %f", architect.alpha_forward)
    root.info("alpha_backward %f", architect.alpha_backward)
    log_value('train_acc', train_acc, epoch)
    root.info('train_acc %f', train_acc)

    # validation
    start_time2 = time.time()
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    end_time2 = time.time()
    root.info("inference time %f", end_time2 - start_time2)
    log_value('valid_acc', valid_acc, epoch)
    root.info('valid_acc %f', valid_acc)

    # test
    start = time.time()
    test_acc, test_obj = infer(test_queue, model, criterion)
    end = time.time()
    root.info("inference time %f", end - start)
    log_value('test_acc', test_acc, epoch)
    root.info('test_acc %f, test_obj %f', test_acc, test_obj)

    # update learning rate
    scheduler.step()

    is_best = valid_acc > best_acc
    best_acc = max(valid_acc, best_acc)
    if is_best:
      root.info('best valid_acc: {} at epoch: {}, test_acc: {}'.format(
        best_acc, epoch, test_acc
      ))
      root.info('Current best genotype = {}'.format(model.genotype()))
      utils.save(model, os.path.join(args.save, 'best_weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
  root = logging.getLogger()

  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  alphas_time = 0
  forward_time = 0
  backward_time = 0
  model.train()
  total_batchs = len(train_queue)
  begin = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    # input, target for weights
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    # input, target for arch
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda(non_blocking=True)
    target_search = target_search.cuda(non_blocking=True)

    # fix weight and update arch
    begin1 = time.time()
    architect.step(input, target, input_search, target_search, lr, optimizer)
    end1 = time.time()
    alphas_time += end1 - begin1

    # fix arch and update weight
    optimizer.zero_grad()
    model.binarization()

    # forward
    begin.record()
    logits = model(input)
    loss = criterion(logits, target)
    end.record()
    forward_time += utils.get_elaspe_time(begin, end)

    # backward
    begin.record()
    loss.backward()
    end.record()
    backward_time += utils.get_elaspe_time(begin, end)

    # update weights
    model.restore()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    # clip arch weigths
    model.clip()

    # track performance
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      root.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      # root.info('step: {:03d} train_labels[0:5]: {}, valid_labels[0:5]: {}'.format(step, target[0:5], target_search[0:5]))
      mapping = {0: 'normal', 1: 'reduce'}
      _step = epoch * total_batchs + step
      for i, arch in enumerate(model.arch_parameters()):
        cell = mapping[i]
        # log_arch(cell, arch, _step)

  return top1.avg, objs.avg, alphas_time, forward_time, backward_time


def infer(valid_queue, model, criterion):
  root = logging.getLogger()

  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  model.binarization()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        root.info('%s %03d %e %f %f', valid_queue.name, step, objs.avg, top1.avg, top5.avg)
        # root.info('step: {:03d} valid_labels[0:5]: {}'.format(step, target[0:5]))
  model.restore()
  return top1.avg, objs.avg


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
  root = logging.getLogger()
  begin = time.time()
  main()
  end = time.time()
  root.info('total search time: {} s'.format(end - begin))
