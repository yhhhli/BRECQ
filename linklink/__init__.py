import os

import torch
import torch.distributed as dist

allreduce = dist.all_reduce
allgather = dist.all_gather
broadcast = dist.broadcast
barrier = dist.barrier
synchronize = torch.cuda.synchronize
init_process_group = dist.init_process_group
get_rank = dist.get_rank
get_world_size = dist.get_world_size


def get_local_rank():
    rank = dist.get_rank()
    return rank % torch.cuda.device_count()


def initialize(backend='nccl', port='2333', job_envrion='normal'):
    """
    Function to initialize distributed enviroments.
    :param backend: nccl backend supports GPU DDP, this should not be modified.
    :param port: port to communication
    :param job_envrion: we refer normal enviroments as the pytorch suggested initialization, the slurm
                        enviroment is used for SLURM job submit system.
    """

    if job_envrion == 'nomal':
        # this step is taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L129
        dist.init_process_group(backend=backend, init_method='tcp://224.66.41.62:23456')
    elif job_envrion == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        if '[' in node_list:
            beg = node_list.find('[')
            pos1 = node_list.find('-', beg)
            if pos1 < 0:
                pos1 = 1000
            pos2 = node_list.find(',', beg)
            if pos2 < 0:
                pos2 = 1000
            node_list = node_list[:min(pos1, pos2)].replace('[', '')
        addr = node_list[8:].replace('-', '.')
        os.environ['MASTER_PORT'] = port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        if backend == 'nccl':
            dist.init_process_group(backend='nccl')
        else:
            dist.init_process_group(backend='gloo', rank=proc_id, world_size=ntasks)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        raise NotImplementedError


def finalize():
    pass


class nn(object):
    SyncBatchNorm2d = torch.nn.BatchNorm2d
    print("You are using fake SyncBatchNorm2d who is actually the official BatchNorm2d")


class syncbnVarMode_t(object):
    L1 = None
    L2 = None

