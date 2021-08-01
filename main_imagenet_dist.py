import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import linklink as link
import time
import hubconf
from quant import *
from linklink.dist_helper import dist_init, allaverage
from data.imagenet import build_imagenet_data
import torch.distributed as dist
import torch.multiprocessing as mp

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def validate_model(test_loader, ann, print_freq=100):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)

    ann.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_iter = len(test_loader)
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.squeeze().view(-1).cuda().long()
            logits = ann(input)

            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss = criterion(logits, target)
            prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))

            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(f'Test: [{i + 1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    top1_sum = torch.Tensor([top1.avg * top1.count]).cuda()
    top5_sum = torch.Tensor([top5.avg * top5.count]).cuda()
    link.allreduce(total_num)
    link.allreduce(loss_sum)
    link.allreduce(top1_sum)
    link.allreduce(top5_sum)
    final_loss = loss_sum.item() / total_num.item()
    final_top1 = top1_sum.item() / total_num.item()
    final_top5 = top5_sum.item() / total_num.item()

    print(f' * Prec@1 {final_top1:.3f}\tPrec@5 {final_top5:.3f}\t\
          Loss {final_loss:.3f}\ttotal_num={total_num.item()}\t')

    return final_top1


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def main_worker(gpu,ngpus_per_node,args):
    # load model
    args.gpu=gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size,rank=args.rank)
    torch.cuda.set_device(args.gpu)
    cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    cnn.cuda()
    cnn.eval()
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
    train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path, dist_sample=(ngpus_per_node > 1))

    cali_data = get_train_samples(train_loader, num_samples=int(args.num_samples / ngpus_per_node))
    torch.backends.cudnn.benchmark = False
    # Initialize weight quantization parameters
    device = next(qnn.parameters()).device
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data[:64].cuda())
    if args.test_before_calibration:
        print('Quantized accuracy before brecq: {}'.format(validate_model(test_loader, qnn)))
    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                  multi_gpu=ngpus_per_node >1)

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)

    # Start calibration
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    print('Weight quantization accuracy: {}'.format(validate_model(test_loader, qnn)))

    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_data[:64].to(device))
        if ngpus_per_node > 1:
            qnn.synchorize_activation_statistics()
        # Disable output quantization because network output
        # does not get involved in further computation

        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p,
                      multi_gpu=ngpus_per_node > 1)
        recon_model(qnn)
        qnn.disable_network_output_quantization()
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,validate_model(test_loader, qnn)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='running parameters',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='', type=str, help='path to ImageNet data', required=True)

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float,
                        help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float,
                        help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')

    # for multi-gpu
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    args = parser.parse_args()
    seed_all(args.seed)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))