import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from tensorboard_logger import configure, log_value

import utils
import genotypes
from model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--valid_batch_size', type=int, default=80, help='valid batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='NASP', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--exec_script', type=str, default='scripts/eval.sh', help='script to run exp')
parser.add_argument('--name', type=str, default="runs", help='name for log')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'), exec_script=args.exec_script)

# Logging configuration
utils.setup_logger(args)

# tensorboard_logger configuration
configure('{}/{}'.format(args.save, args.name))


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

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  root.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  # Data loading code
  train_queue, train_sampler, valid_queue = get_train_validation_loader(args)
  test_queue = get_test_loader(args)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_acc = 0
  for epoch in range(args.epochs):
    lr = scheduler.get_lr()[0]
    log_value("lr", lr, epoch)
    root.info('epoch %d lr %e', epoch, lr)
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    # train
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    root.info('train_acc %f, train_loss %f', train_acc, train_obj)
    log_value('train_acc', train_acc, epoch)

    # valid
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    root.info('valid_acc %f, valid_obj %f', valid_acc, valid_obj)
    log_value('valid_acc', valid_acc, epoch)

    # test
    test_acc, test_obj = infer(test_queue, model, criterion)
    root.info('test_acc %f, test_obj %f', test_acc, test_obj)
    log_value('test_acc', test_acc, epoch)

    # update learning rate
    scheduler.step()
    utils.save(model, os.path.join(args.save, 'weights.pt'))

    is_best = valid_acc > best_acc
    best_acc = max(valid_acc, best_acc)
    if is_best:
      root.info('best test_acc: {} at Epoch {} when valid_acc is {}'.format(
        test_acc, epoch, best_acc
      ))


def train(train_queue, model, criterion, optimizer):
  root = logging.getLogger()
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      root.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  root = logging.getLogger()
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        root.info('%s %03d %e %f %f', valid_queue.name, step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def get_train_validation_loader(args):
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  # train[0:split] as training data
  train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.train_batch_size,
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

  # train[split:] as validation data
  valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
  valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.valid_batch_size,
    num_workers=args.workers, pin_memory=True, sampler=valid_sampler)
  valid_queue.name = 'valid'

  return train_queue, train_sampler, valid_queue


def get_test_loader(args):
  _, test_transform = utils._data_transforms_cifar10(args)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
  test_queue = torch.utils.data.DataLoader(
    test_data, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
  test_queue.name = 'test'
  return test_queue


if __name__ == '__main__':
  root = logging.getLogger()
  begin = time.time()
  main()
  end = time.time()
  root.info('total train time: {} s'.format(end - begin))

