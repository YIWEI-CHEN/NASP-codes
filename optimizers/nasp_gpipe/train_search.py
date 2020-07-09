import argparse
import glob
import json
import logging
from collections import OrderedDict, Iterator
from typing import Tuple

import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.autograd import Variable
from torchgpipe import GPipe


sys.path.insert(0, '../../')
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import NasbenchWrapper

from optimizers.darts import utils
from optimizers.nasp_gpipe.architect import ArchitectNASP as Architect
from optimizers.nasp_gpipe.model_search import NASPNetwork as Network
from optimizers.nasp_gpipe import update_type

from optimizers.analyze import Analyzer
from copy import deepcopy
from numpy import linalg as LA

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the darts corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='2', help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=9, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random_ws seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training darts')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
parser.add_argument('--warm_start_epochs', type=int, default=0,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--chunks', type=int, default=2, help='chunk size')
args = parser.parse_args()

args.save = '../../experiments/nasp_gpipe/search_space_{}/search-{}-{}-{}-{}'.format(
    args.search_space, args.save, time.strftime("%Y%m%d-%H%M%S"), args.seed, args.search_space)
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
if not args.arch_weight_decay == 1e-3:
    args.save += '-alpha_l2-' + str(args.arch_weight_decay)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
args.save += '-' + str(np.random.randint(10000))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# Dump the config of the run
with open(os.path.join(args.save, 'config.json'), 'w') as fp:
    json.dump(args.__dict__, fp)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')

CIFAR_CLASSES = 10

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
num_gpus = len(args.gpu.split(','))

def flatten_sequential(module: nn.Sequential) -> nn.Sequential:
  """flatten_sequentials a nested sequential module."""
  d = OrderedDict(_flatten_sequential(module))
  return nn.Sequential(d)


def _flatten_sequential(module):
  for name, child in module.named_children():
    # flatten_sequential child sequential layers only.
    if isinstance(child, nn.Sequential):
      for sub_name, sub_child in _flatten_sequential(child):
        yield (f'{name}_{sub_name}', sub_child)
    else:
      yield (name, child)


class MyGPipe(GPipe):
  def __init__(self, module, balance,
               devices = None, chunks = 1, checkpoint = 'except_last', deferred_batch_norm = False):
    seq_module = flatten_sequential(module)
    super().__init__(seq_module, balance,
                     devices=module.devices, chunks=chunks, checkpoint=checkpoint, deferred_batch_norm=deferred_batch_norm)
    self._module = module
    self.len_arch_param = module.len_arch_param
    self.devices = module.devices

  def binarization(self):
    self._module.binarization()

  def restore(self):
    self._module.restore()

  def genotype(self):
    return self._module.genotype()

  def clip(self):
    self._module.clip()

  def arch_parameters(self):
    return self._module.arch_parameters()


def main():
    if not 'debug' in args.save:
        from nasbench_analysis import eval_darts_one_shot_model_in_nasbench as naseval
    # Select the search space to search in
    if args.search_space == '1':
        search_space = SearchSpace1()
    elif args.search_space == '2':
        search_space = SearchSpace2()
    elif args.search_space == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')
    
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    if num_gpus > 1:
        balance = [(args.layers + 3) // num_gpus for _ in range(num_gpus)]
        balance[0] += 1
        balance[-1] += 2
    else:
        balance = [2 + args.layers + 4]
    root.info("balanace {}".format(balance))
    chunks = args.chunks

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, 
                    output_weights=args.output_weights, steps=search_space.num_intermediate_nodes, 
                    search_space=search_space, gpus=args.gpu)
    # model.to(torch.device('cuda:0}'))
    model = MyGPipe(model, balance, chunks=chunks)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    
    if 'debug' in args.save:
        split = args.batch_size
        num_train = 2 * args.batch_size

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    lr_multiplier = max(1.0, args.batch_size / 96)
    warmup = 4.0
    decay = 0.5

    def gradual_warmup_linear_scaling(step: int) -> float:
        epoch = step / float(args.epochs)

        # Gradual warmup
        warmup_ratio = min(warmup, epoch) / warmup
        multiplier = warmup_ratio * (lr_multiplier - 1.0) + 1.0

        if step < 17:
            return 1.0 * multiplier
        elif step < 33:
            return decay * multiplier
        elif step < 44:
            return decay ** 2 * multiplier
        return decay ** 3 * multiplier

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=gradual_warmup_linear_scaling)

    analyzer = Analyzer(model, args)
    architect = Architect(model, criterion, args)

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        # increase the cutout probability linearly throughout search
        train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
        logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                     train_transform.transforms[-1].cutout_prob)

        # Save the one shot model architecture weights for later analysis
        arch_filename = os.path.join(args.save, 'one_shot_architecture_{}.obj'.format(epoch))
        with open(arch_filename, 'wb') as filehandler:
            numpy_tensor_list = []
            dst_list = [0 for _ in range(model.len_arch_param)]
            # arch_params = model.arch_parameters()
            for l in range(args.layers):
                arch_params = model.arch_parameters()[l * model.len_arch_param: (l + 1) * model.len_arch_param]
                for i, tensor in enumerate(arch_params):
                    dst_list[i] += tensor.detach().cpu().numpy()
            for tensor in dst_list:
                numpy_tensor_list.append(tensor / args.layers)
            pickle.dump(numpy_tensor_list, filehandler)

        # # Save the entire one-shot-model
        # filepath = os.path.join(args.save, 'one_shot_model_{}.obj'.format(epoch))
        # torch.save(model.state_dict(), filepath)

        for i in numpy_tensor_list:
            logging.info(i)

        # training
        train_acc, train_obj, ev = train(train_queue, valid_queue, model, architect, criterion, 
                                         optimizer, lr, epoch, analyzer)
        logging.info('train_acc %f', train_acc)
        logging.info('eigenvalue %f', ev)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)
        writer.add_scalar('Analysis/eigenvalue', ev, epoch)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('Acc/valid', valid_acc, epoch)
        writer.add_scalar('Obj/valid', valid_obj, epoch)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

        if not 'debug' in args.save:
            # benchmark
            logging.info('STARTING EVALUATION')
            test, valid, runtime, params = naseval.eval_one_shot_model(
                config=args.__dict__, model=arch_filename)
            
            index = np.random.choice(list(range(3)))
            test, valid, runtime, params = np.mean(test), np.mean(valid), np.mean(runtime), np.mean(params)
            logging.info('TEST ERROR: %.3f | VALID ERROR: %.3f | RUNTIME: %f | PARAMS: %d'
                        % (test, valid, runtime, params))
            writer.add_scalar('Analysis/test', test, epoch)
            writer.add_scalar('Analysis/valid', valid, epoch)
            writer.add_scalar('Analysis/runtime', runtime, epoch)
            writer.add_scalar('Analysis/params', params, epoch)
        scheduler.step()
    writer.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, analyzer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    first_device = model.devices[0]
    last_device = model.devices[-1]

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(first_device, non_blocking=True)
        target = target.cuda(last_device, non_blocking=True)

        # get a minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        input_search = input_search.cuda(first_device, non_blocking=True)
        target_search = target_search.cuda(last_device, non_blocking=True)

        # Allow for warm starting of the one-shot model for more reliable architecture updates.
        if epoch >= args.warm_start_epochs:
            architect.step(input, target, input_search, target_search, lr, optimizer)

        optimizer.zero_grad()
        model.binarization()
        update_type.set_weigths()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        model.restore()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.clip()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break
            
    # analyze
    # _data_loader = deepcopy(train_queue)
    # input, target = next(iter(_data_loader))

    # input = Variable(input, requires_grad=False).cuda()
    # target = Variable(target, requires_grad=False).cuda(non_blocking=True)
    
    # model.binarization()
    # H = analyzer.compute_Hw(input, target, input_search, target_search,
    #                         lr, optimizer, False)
    # model.restore()
    # g = analyzer.compute_dw(input, target, input_search, target_search,
    #                         lr, optimizer, False)
    # g = torch.cat([x.view(-1) for x in g])

    # del _data_loader
    
    # ev = max(LA.eigvals(H.cpu().data.numpy()))
    # ev = np.linalg.norm(ev)
    ev = 0

    return top1.avg, objs.avg, ev


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    model.binarization()

    first_device = model.devices[0]
    last_device = model.devices[-1]

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(first_device, non_blocking=True)
            target = target.cuda(last_device, non_blocking=True)

            update_type.set_weigths()
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                if 'debug' in args.save:
                    break
    model.restore()
    return top1.avg, objs.avg


if __name__ == '__main__':
    root = logging.getLogger()
    begin = time.time()
    main()
    end = time.time()
    root.info('total search time: {} s'.format(end - begin))
