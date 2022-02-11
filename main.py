import argparse
import gc
import random
import re
import os
import time
import math
import bisect
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np

import core.norm_dist
from utils import random_seed, create_result_dir, Logger, TableLogger, AverageMeter
from attack import AttackPGD
from ell_inf_models import *
from core.modules import NormDistBase
from torch.nn.functional import cross_entropy
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter

from torch.autograd import gradcheck
import torch

parser = argparse.ArgumentParser(description='L-infinity Dist Net')

parser.add_argument("--manual-result-dir", default="result/CIFAR10_2", type=str)

parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--model', default='MLPModel(depth=6,width=5120,identity_val=10.0, scalar=True)', type=str)
parser.add_argument('--loss', default='mixture', type=str)

parser.add_argument('--p-start', default=8.0, type=float)
parser.add_argument('--p-end', default=1000.0, type=float)

parser.add_argument('--eps-train', default=None, type=float)
parser.add_argument('--eps-test', default=None, type=float)
parser.add_argument('--eps-smooth', default=0, type=float)

parser.add_argument('--epochs', default='0,0,100,1250,1300', type=str)
# corresponding to: eps_start, eps_end, p_start, p_end, total
parser.add_argument('--decays', default=None, type=str)
parser.add_argument('-b', '--batch-size', default=512, type=int)
parser.add_argument('--lr', default=0.03, type=float)
parser.add_argument('--scalar-lr', default=0.006, type=float)

parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.99, type=float)
parser.add_argument('--epsilon', default=1e-10, type=float)

parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--checkpoint', default=None, type=str)

parser.add_argument('--gpu', default=-1, type=int, help='GPU id to use')
parser.add_argument('--dist-url', default='tcp://localhost:23456')
parser.add_argument('--world-size', default=1)
parser.add_argument('--rank', default=0)

parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency')
parser.add_argument('--innerb', default=None, type=int, help='inner batch size used by model')
parser.add_argument('--saving-epoch', default=None, type=int, help='every epochs save state of the model')
parser.add_argument('--result-dir', default='result', type=str)
parser.add_argument('--filter-name', default='', type=str)
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--visualize', action='store_true')


def cal_acc_cert(outputs, targets, train_eps, test_eps):
    outputs *= (2 * train_eps)
    outputs = outputs.scatter(1, targets.view(-1, 1), 2 * (train_eps - test_eps))
    predicted = torch.max(outputs.data, 1)[1]
    return (predicted == targets).float().mean().item()


def cal_acc(outputs, targets):
    predicted = torch.max(outputs.data, 1)[1]
    return (predicted == targets).float().mean().item()


def parallel_reduce(*argv):
    tensor = torch.FloatTensor(argv).cuda()
    torch.distributed.all_reduce(tensor)
    ret = tensor.cpu() / torch.distributed.get_world_size()
    return ret.tolist()


@contextmanager
def eval(model):
    state = [m.training for m in model.modules()]
    model.eval()
    yield
    for m, s in zip(model.modules(), state):
        m.train(s)


def train(net, up, down, loss_fun, epoch, train_loader, optimizer, schedule, logger, train_logger, gpu, parallel,
          print_freq, test_eps, inner_batch_size):
    batch_time, losses, correct_accs, certified_accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    epoch_start_time = start
    train_loader_len = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        step = batch_idx + train_loader_len * epoch
        if batch_idx % 100 == 0:
            get_model_detail(net, step)

        eps, p, mix, lr = schedule(epoch, batch_idx)
        writer.add_scalar("p", p, step)
        print(f"\r step={batch_idx} p={round(p, 4)} ", end=" ")
        print(round(torch.max(inputs).item(), 2), round(torch.min(inputs).item(), 2), end=" ")
        inputs = inputs.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        optimizer.zero_grad()
        iteration_num = inputs.shape[0] // inner_batch_size
        for i in range(0, iteration_num):
            l = i * inner_batch_size
            r = l + inner_batch_size
            outputs, worst_outputs = net(inputs[l:r], targets=targets[l:r], eps=eps, up=up, down=down)
            loss = loss_fun(outputs, worst_outputs, targets[l:r])/iteration_num
            with torch.no_grad():
                losses.update(loss.data.item()*iteration_num, targets[l:r].size(0))
                correct_accs.update(cal_acc(outputs.data, targets[l:r]), targets[l:r].size(0))
                certified_accs.update(cal_acc_cert(worst_outputs.data, targets[l:r], eps, test_eps),
                                      targets[l:r].size(0))

            loss.backward()
            gc.collect()
            torch.cuda.empty_cache()

        optimizer.step()

        batch_time.update(time.time() - start)
        if (batch_idx + 1) % print_freq == 0 and logger is not None:
            writer.add_scalar("training loss", losses.queueavg, step)
            writer.add_scalar("train accuracy", correct_accs.queueavg, step)
            writer.add_scalar("certified accuracy", certified_accs.queueavg, step)

            logger.print('Epoch: [{0}][{1}/{2}]   '
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                         'lr {lr:.4f}   p {p:.2f}   eps {eps:.4f}   mix {mix:.4f}   '
                         'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                         'Acc {acc.val:.4f} ({acc.avg:.4f})   '
                         'Certified (not fake) {cert.val:.4f} ({cert.avg:.4f})'.format(
                epoch, batch_idx + 1, train_loader_len, batch_time=batch_time,
                lr=lr, p=p, eps=eps, mix=mix, loss=losses, acc=correct_accs, cert=certified_accs))
        start = time.time()

    step = train_loader_len * (epoch + 1)
    loss, acc, cert = losses.avg, correct_accs.avg, certified_accs.avg
    if parallel:
        loss, acc, cert = parallel_reduce(loss, acc, cert)
    if train_logger is not None:
        train_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc, 'certified': cert})
    if logger is not None:
        eps, p, mix, lr = schedule(epoch, 0)
        logger.print('Epoch {0}:  train loss {loss:.4f}   train acc {acc:.4f}   worst(cert) {cert:.4f}   '
                     'lr {lr:.4f}   p {p:.2f}   eps {eps:.4f}   mix {mix:.4f}   time {time:.2f}'.format(
            epoch, loss=loss, acc=acc, cert=cert, lr=lr, p=p, eps=eps, mix=mix,
            time=time.time() - epoch_start_time))
    return loss, acc, cert


@torch.no_grad()
def test(net, epoch, test_loader, logger, test_logger, gpu, parallel, print_freq, test_eps):
    batch_time, accs, certified_accs = [AverageMeter() for _ in range(3)]
    start = time.time()
    epoch_start_time = start
    test_loader_len = len(test_loader)

    with eval(net):
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            outputs, worst_outputs = net(inputs, targets=targets, eps=test_eps, up=None, down=None)
            accs.update(cal_acc(outputs, targets), targets.size(0))
            certified_accs.update(cal_acc(worst_outputs.data, targets), targets.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            if (batch_idx + 1) % print_freq == 0 and logger is not None:
                logger.print('Test: [{0}/{1}]   '
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                             'Acc {acc.val:.4f} ({acc.avg:.4f})'
                             'Certified (not fake) {cert.val:.4f} ({cert.avg:.4f})'.format(
                    batch_idx + 1, test_loader_len, batch_time=batch_time, acc=accs, cert=certified_accs))

    acc = accs.avg
    cert_acc = certified_accs.avg
    if parallel:
        acc = parallel_reduce(accs.avg)
    if test_logger is not None:
        test_logger.log({'epoch': epoch, 'acc': acc})
    if logger is not None:
        elapse = time.time() - epoch_start_time
        logger.print(
            'Epoch %d:  ' % epoch + 'test acc ' + f'{acc:.4f}' + '  test cert acc ' + f'{cert_acc:.4f}' + '  time ' + f'{elapse:.2f}')
    return acc


def gen_adv_examples(net, attacker, test_loader, gpu, parallel, logger, fast=False, very_fast=True):
    correct = 0
    tot_num = 0
    size = len(test_loader)

    with eval(net):
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            result = torch.ones(targets.size(0), dtype=torch.bool, device=targets.device)
            for i in range(1):
                perturb = attacker.find(inputs, targets)
                with torch.no_grad():
                    outputs = net(perturb)
                    predicted = torch.max(outputs.data, 1)[1]
                    result &= (predicted == targets)
            correct += result.float().sum().item()
            tot_num += inputs.size(0)
            print(f"\r {batch_idx}/{len(test_loader)}  PGD acc:{round(correct / tot_num, 3)} ")
            if very_fast and tot_num > 50:
                break
            if fast and batch_idx * 10 >= size:
                break

    acc = correct / tot_num * 100
    if parallel:
        acc, = parallel_reduce(acc)
    if logger is not None:
        logger.print('adversarial attack acc ' + f'{acc:.4f}')
    return acc


@torch.no_grad()
def certified_test(net, eps, up, down, epoch, test_loader, logger, certified_logger, gpu, parallel):
    outputs = []
    worst_outputs = []
    labels = []
    normdist_models = get_normdist_models(net)
    cur_p = [m.p for m in normdist_models]
    for m in normdist_models:
        m.p = float('inf')

    with eval(net):
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            output, worst = net(inputs, targets=targets, eps=eps, up=up, down=down)
            outputs.append(output)
            worst_outputs.append(worst)
            labels.append(targets)
    outputs = torch.cat(outputs, dim=0)
    worst_outputs = torch.cat(worst_outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    correct = cal_acc(outputs, labels)
    certified = cal_acc(worst_outputs, labels)

    if parallel:
        correct, certified = parallel_reduce(correct, certified)
    if logger is not None:
        logger.print('Epoch %d: ' % epoch + ' clean acc ' + f'{correct:.4f}' +
                     '   certified acc ' + f'{certified:.4f}')
    if certified_logger is not None:
        certified_logger.log({'epoch': epoch, 'acc': correct, 'certified': certified})
    for m, p in zip(normdist_models, cur_p):
        m.p = p
    return correct, certified


def parse_function_call(s):
    s = re.split(r'[()]', s)
    if len(s) == 1:
        return s[0], {}
    name, params, _ = s
    params = re.split(r',\s*', params)
    params = dict([p.split('=') for p in params])
    for key, value in params.items():
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                special = {'True': True, 'False': False, 'None': None}
                try:
                    params[key] = special[value]
                except KeyError:
                    pass
    return name, params


def get_normdist_models(model):
    return [m for m in model.modules() if isinstance(m, NormDistBase)]


def create_schedule(args, batch_per_epoch, model, optimizer, loss, eps_schedule='linear'):
    print(f"batch per epoch: {batch_per_epoch}")
    epoch_eps_start, epoch_eps_end, epoch_p_start, epoch_p_end, epoch_tot = args.epochs
    if args.decays is not None:
        decays = [int(epoch) for epoch in args.decays.split(',')]
    else:
        decays = None
    speed_p = math.log(args.p_end / args.p_start)
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    smooth_r = args.eps_smooth

    def num_batches(epoch, minibatch=0):
        return epoch * batch_per_epoch + minibatch

    def cal_ratio(epoch, epoch_start, epoch_end, minibatch):
        if epoch_end <= epoch_start:
            return 1
        return min(max(num_batches(epoch - epoch_start, minibatch) / num_batches(epoch_end - epoch_start), 0), 1)

    def schedule(epoch, minibatch):
        if decays is None:
            ratio = 0  # cal_ratio(epoch, 0, epoch_tot, minibatch)
            for param_group, lr in zip(optimizer.param_groups, lrs):
                param_group['lr'] = 0.5 * lr * (1 + math.cos((ratio * math.pi)))
        else:
            index = bisect.bisect_right(decays, epoch)
            for param_group, lr in zip(optimizer.param_groups, lrs):
                param_group['lr'] = lr / (5 ** index)

        ratio = cal_ratio(epoch, epoch_p_start, epoch_p_end, minibatch)
        if ratio >= 1 and args.p_end >= 100:
            p_norm = float('inf')
        else:
            p_norm = args.p_start * math.exp(speed_p * ratio)
        for m in get_normdist_models(model):
            m.p = p_norm

        ratio = cal_ratio(epoch, epoch_eps_start, epoch_eps_end, minibatch)
        if eps_schedule == 'linear' or smooth_r == 0:
            cur_eps = args.eps_train * ratio
        elif eps_schedule == 'smooth':
            k = 1 / ((4 - 3 * smooth_r) * smooth_r ** 3)
            if ratio < smooth_r:
                cur_eps = k * ratio ** 4 * args.eps_train
            else:
                cur_eps = ((1 - k * smooth_r ** 4) / (1 - smooth_r) * (ratio - 1) + 1) * args.eps_train
        else:
            raise NotImplementedError

        ratio = cal_ratio(epoch, epoch_p_start, epoch_p_end, minibatch)
        if hasattr(loss, 'update'):
            loss.update(ratio)
            lam = loss.lam
        else:
            lam = 0

        return cur_eps, p_norm, lam, optimizer.param_groups[0]['lr']

    return schedule


class hinge():
    def __init__(self, mix=0.25):
        self.mix = mix

    def __call__(self, outputs, worst_outputs, targets):
        res = worst_outputs.clamp(min=0)
        return (1 - self.mix) * res.max(dim=1)[0].mean() + self.mix * res.mean()


class crossentropy():
    def __init__(self, mix=0):
        self.mix = mix

    def __call__(self, outputs, worst_outputs, targets):
        return (1 - self.mix) * cross_entropy(worst_outputs, targets) + self.mix * cross_entropy(outputs, targets)


class mixture():
    def __init__(self, lam0=0.1, lam_end=0.001, clip=1):
        self.lam_start = lam0
        self.lam_end = lam_end
        self.lam = lam0
        self.speed = math.log(self.lam_end / self.lam_start)
        self.clip = clip

    def update(self, ratio):
        self.lam = self.lam_start * math.exp(self.speed * ratio)

    def __call__(self, outputs, worst_outputs, targets):
        res = worst_outputs.clamp(min=0, max=self.clip)
        return res.max(dim=1)[0].mean() + self.lam * cross_entropy(outputs, targets)


def get_model_detail(model, step):
    for name, p in model.named_parameters():
        if name.endswith(".r"):
            for q in range(0, 10, 1):
                writer.add_scalar(f"detail/{name}/{q * 10}/distribution", torch.quantile(p, q / 10).item(), step)

    for name, p in model.named_parameters():
        if name.endswith(".imp"):
            p = torch.nn.Softmax(dim=1)(p)
            if p.shape[1] > 20000 // p.shape[0]:
                p = p[:20000 // p.shape[0]]

            for i in range(5):
                writer.add_scalar(f"detail/{name}/{20 * i}/distribution", torch.quantile(p, 0.2 * i).item(), step)
            for i in range(10):
                writer.add_scalar(f"detail/{name}/{80 + 2 * i}/distribution", torch.quantile(p, 0.8 + 0.02 * i).item(),
                                  step)


class normal_hinge():
    def __init__(self, mix=0.25):
        pass

    def __call__(self, outputs, worst_outputs, targets):
        res = worst_outputs.clamp(min=0)
        return res.mean()


def main_worker(gpu, model_dict, parallel, args, result_dir):
    if parallel:
        args.rank = args.rank + gpu
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
    torch.backends.cudnn.benchmark = True
    random_seed(args.seed + args.rank)  # make data aug different for different processes
    torch.cuda.set_device(gpu)

    assert args.batch_size % args.world_size == 0
    from dataset import load_data, get_statistics, default_eps, input_dim
    train_loader, test_loader = load_data(args.dataset, 'data/', args.batch_size // args.world_size, parallel,
                                          augmentation=True)
    mean, std = get_statistics(args.dataset)
    num_classes = len(train_loader.dataset.classes)

    model_name, params = parse_function_call(args.model)

    model = globals()[model_name](model_dict=model_dict, input_dim=input_dim[args.dataset], num_classes=num_classes,
                                  **params)
    model = model.cuda(gpu)
    if parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.eps_test is None:
        args.eps_test = default_eps[args.dataset]
    if args.eps_train is None:
        args.eps_train = args.eps_test

    loss_name, params = parse_function_call(args.loss)
    if loss_name == 'cross_entropy':
        loss = cross_entropy
    elif loss_name == 'hinge':
        loss = normal_hinge()
    else:
        loss = globals()[loss_name](**params)

    output_flag = not parallel or gpu == 0
    if output_flag:
        logger = Logger(os.path.join(result_dir, 'log.txt'))
        for arg in vars(args):
            logger.print(arg, '=', getattr(args, arg))
        logger.print(train_loader.dataset.transform)
        logger.print(model)
        logger.print('number of params: ', sum([p.numel() for p in model.parameters()]))
        train_logger = TableLogger(os.path.join(result_dir, 'train.log'), ['epoch', 'loss', 'acc', 'certified'])
        test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['epoch', 'acc'])
        train_inf_logger = TableLogger(os.path.join(result_dir, 'train_inf.log'), ['epoch', 'acc', 'certified'])
        test_inf_logger = TableLogger(os.path.join(result_dir, 'test_inf.log'), ['epoch', 'acc', 'certified'])
    else:
        logger = train_logger = test_logger = train_inf_logger = test_inf_logger = None

    args.eps_train /= std
    args.eps_test /= std

    params = [
        {'params': [p for name, p in model.named_parameters() if 'scalar' not in name], 'lr': args.lr},
        {'params': [p for name, p in model.named_parameters() if 'scalar' in name], 'lr': args.scalar_lr},
    ]
    optimizer = Adam(params, betas=(args.beta1, args.beta2), eps=args.epsilon)

    if args.checkpoint:
        assert os.path.isfile(args.checkpoint)
        if parallel:
            torch.distributed.barrier()
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(gpu))
        state_dict = checkpoint['state_dict']
        if next(iter(state_dict)).startswith('module.') and not parallel:
            new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
            state_dict = new_state_dict
        elif not next(iter(state_dict)).startswith('module.') and parallel:
            new_state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded '{}'".format(args.checkpoint))
        if parallel:
            torch.distributed.barrier()

    up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).cuda(gpu)
    down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).cuda(gpu)
    attacker = AttackPGD(model, args.eps_test, step_size=args.eps_test / 4, num_steps=100, up=up, down=down)
    args.epochs = [int(epoch) for epoch in args.epochs.split(',')]
    schedule = create_schedule(args, len(train_loader), model, optimizer, loss, 'smooth')

    for epoch in range(args.start_epoch, args.epochs[-1]):

        print(f"epoch = {epoch}")
        if parallel:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc, train_cert = train(model, None, None, loss, epoch, train_loader, optimizer, schedule,
                                                  logger, train_logger, gpu, parallel, args.print_freq, args.eps_test,
                                                  args.innerb)

        if (epoch + 1) % args.saving_epoch == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(result_dir, f"model{epoch}.pth"))

        if epoch % 1 == 0 or epoch >= args.epochs[-1] - 5:
            test_acc = test(model, epoch, test_loader, logger, test_logger, gpu, parallel, args.print_freq,
                            args.eps_test)
            if epoch % 8 == 9:
                if logger is not None:
                    logger.print('Calculating metrics for L_infinity dist model on training set')
                train_inf_acc, train_inf_cert = certified_test(model, args.eps_test, None, None, epoch, train_loader,
                                                               logger, train_inf_logger, gpu, parallel)

            if logger is not None:
                logger.print('Calculating metrics for L_infinity dist model on test set')
            test_inf_acc, test_inf_cert = certified_test(model, args.eps_test, None, None, epoch, test_loader,
                                                         logger, test_inf_logger, gpu, parallel)
            if writer is not None:
                writer.add_scalar('test acc', test_acc, epoch)
                if epoch % 8 == 7:
                    writer.add_scalar('train acc (inf model)', train_inf_acc, epoch)
                    writer.add_scalar('train certified acc (inf model)', train_inf_cert, epoch)
                writer.add_scalar('test acc (inf model)', test_inf_acc, epoch)
                writer.add_scalar('test certified acc (inf model)', test_inf_cert, epoch)
        if epoch >= args.epochs[-1] * 0.9 and (epoch % 50 == 49 or epoch >= args.epochs[-1] - 5):
            if logger is not None:
                logger.print('Generate adversarial examples on test dataset')
            robust_test_acc = gen_adv_examples(model, attacker, test_loader, gpu, parallel, logger, fast=False)
            if writer is not None:
                writer.add_scalar('robust test acc(gen adv examples)', robust_test_acc, epoch)
    schedule(args.epochs[-1], 0)
    logger.print('============Training completes===========')
    if logger is not None:
        logger.print('Generate adversarial examples on test dataset')
    gen_adv_examples(model, attacker, test_loader, gpu, parallel, logger, fast=False)
    if logger is not None:
        logger.print('Calculating test acc and certified test acc')
    certified_test(model, args.eps_test, None, None, args.epochs[-1], test_loader,
                   logger, test_inf_logger, gpu, parallel)
    if output_flag:
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(result_dir, 'model.pth'))
    if writer is not None:
        writer.close()


def main(father_handle, **extra_argv):
    ###################################################
    ##############################################################################
    run_name = "exp12"
    model_dict = {'learnable length': True, 'learnable r': True, 'initial r': 4}
    ##############################################################################
    ####################################################

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    for key, val in extra_argv.items():
        setattr(args, key, val)

    if args.manual_result_dir == None:
        result_dir = create_result_dir(args)
    else:
        result_dir = args.manual_result_dir

    result_dir += "lenght_" + str(model_dict['learnable length']) + "_r" + str(model_dict['learnable r']) + "_" + str(
        model_dict['initial r']) + run_name
    print(result_dir)
    global writer
    writer = SummaryWriter(log_dir=result_dir, flush_secs=30)

    writer.add_custom_scalars(layout={"imp's": {"layer0": ['Multiline', ['detail/fc_dist.0.imp/0/distribution',
                                                                         'detail/fc_dist.0.imp/20/distribution',
                                                                         'detail/fc_dist.0.imp/40/distribution',
                                                                         'detail/fc_dist.0.imp/60/distribution',
                                                                         'detail/fc_dist.0.imp/80/distribution',
                                                                         'detail/fc_dist.0.imp/82/distribution',
                                                                         'detail/fc_dist.0.imp/84/distribution',
                                                                         'detail/fc_dist.0.imp/86/distribution',
                                                                         'detail/fc_dist.0.imp/88/distribution',
                                                                         'detail/fc_dist.0.imp/90/distribution',
                                                                         'detail/fc_dist.0.imp/92/distribution',
                                                                         'detail/fc_dist.0.imp/94/distribution',
                                                                         'detail/fc_dist.0.imp/96/distribution',
                                                                         'detail/fc_dist.0.imp/98/distribution']],
                                                "layer1": ['Multiline', ['detail/fc_dist.1.imp/0/distribution',
                                                                         'detail/fc_dist.1.imp/20/distribution',
                                                                         'detail/fc_dist.1.imp/40/distribution',
                                                                         'detail/fc_dist.1.imp/60/distribution',
                                                                         'detail/fc_dist.1.imp/80/distribution',
                                                                         'detail/fc_dist.1.imp/82/distribution',
                                                                         'detail/fc_dist.1.imp/84/distribution',
                                                                         'detail/fc_dist.1.imp/86/distribution',
                                                                         'detail/fc_dist.1.imp/88/distribution',
                                                                         'detail/fc_dist.1.imp/90/distribution',
                                                                         'detail/fc_dist.1.imp/92/distribution',
                                                                         'detail/fc_dist.1.imp/94/distribution',
                                                                         'detail/fc_dist.1.imp/96/distribution',
                                                                         'detail/fc_dist.1.imp/98/distribution']],
                                                "layer2": ['Multiline', ['detail/fc_dist.2.imp/0/distribution',
                                                                         'detail/fc_dist.2.imp/20/distribution',
                                                                         'detail/fc_dist.2.imp/40/distribution',
                                                                         'detail/fc_dist.2.imp/60/distribution',
                                                                         'detail/fc_dist.2.imp/80/distribution',
                                                                         'detail/fc_dist.2.imp/82/distribution',
                                                                         'detail/fc_dist.2.imp/84/distribution',
                                                                         'detail/fc_dist.2.imp/86/distribution',
                                                                         'detail/fc_dist.2.imp/88/distribution',
                                                                         'detail/fc_dist.2.imp/90/distribution',
                                                                         'detail/fc_dist.2.imp/92/distribution',
                                                                         'detail/fc_dist.2.imp/94/distribution',
                                                                         'detail/fc_dist.2.imp/96/distribution',
                                                                         'detail/fc_dist.2.imp/98/distribution']]},
                                      "r's": {"layer0": ['Multiline', ['detail/fc_dist.0.r/0/distribution',
                                                                       'detail/fc_dist.0.r/10/distribution',
                                                                       'detail/fc_dist.0.r/20/distribution',
                                                                       'detail/fc_dist.0.r/30/distribution',
                                                                       'detail/fc_dist.0.r/40/distribution',
                                                                       'detail/fc_dist.0.r/50/distribution',
                                                                       'detail/fc_dist.0.r/60/distribution',
                                                                       'detail/fc_dist.0.r/70/distribution',
                                                                       'detail/fc_dist.0.r/80/distribution',
                                                                       'detail/fc_dist.0.r/90/distribution']],
                                              "layer1": ['Multiline', ['detail/fc_dist.1.r/0/distribution',
                                                                       'detail/fc_dist.1.r/10/distribution',
                                                                       'detail/fc_dist.1.r/20/distribution',
                                                                       'detail/fc_dist.1.r/30/distribution',
                                                                       'detail/fc_dist.1.r/40/distribution',
                                                                       'detail/fc_dist.1.r/50/distribution',
                                                                       'detail/fc_dist.1.r/60/distribution',
                                                                       'detail/fc_dist.1.r/70/distribution',
                                                                       'detail/fc_dist.1.r/80/distribution',
                                                                       'detail/fc_dist.1.r/90/distribution']],
                                              "layer2": ['Multiline', ['detail/fc_dist.2.r/0/distribution',
                                                                       'detail/fc_dist.2.r/10/distribution',
                                                                       'detail/fc_dist.2.r/20/distribution',
                                                                       'detail/fc_dist.2.r/30/distribution',
                                                                       'detail/fc_dist.2.r/40/distribution',
                                                                       'detail/fc_dist.2.r/50/distribution',
                                                                       'detail/fc_dist.2.r/60/distribution',
                                                                       'detail/fc_dist.2.r/70/distribution',
                                                                       'detail/fc_dist.2.r/80/distribution',
                                                                       'detail/fc_dist.2.r/90/distribution']]}})

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if father_handle is not None:
        father_handle.put(result_dir)
    if args.gpu != -1:
        main_worker(args.gpu, model_dict, False, args, result_dir)
    else:
        n_procs = torch.cuda.device_count()
        args.world_size *= n_procs
        args.rank *= n_procs
        torch.multiprocessing.spawn(main_worker, nprocs=n_procs, args=(True, args, result_dir))


if __name__ == '__main__':
    main(None)
