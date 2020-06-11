import torch
import torch.distributed as dist
import pickle


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def can_log():
    return is_main_process()


def dist_print(*args, **kwargs):
    if can_log():
        print(*args, **kwargs)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def dist_cat_reduce_tensor(tensor):
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor
    # dist_print(tensor)
    rt = tensor.clone()
    all_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(all_list,rt)
    # dist_print(all_list[0][1],all_list[1][1],all_list[2][1],all_list[3][1])
    # dist_print(all_list[0][2],all_list[1][2],all_list[2][2],all_list[3][2])
    # dist_print(all_list[0][3],all_list[1][3],all_list[2][3],all_list[3][3])
    # dist_print(all_list[0].shape)
    return torch.cat(all_list,dim = 0)

def dist_sum_reduce_tensor(tensor):
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt


def dist_mean_reduce_tensor(tensor):
    rt = dist_sum_reduce_tensor(tensor)
    rt /= get_world_size()
    return rt


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


from torch.utils.tensorboard import SummaryWriter


class DistSummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        if can_log():
            super(DistSummaryWriter, self).__init__(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        if can_log():
            super(DistSummaryWriter, self).add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if can_log():
            super(DistSummaryWriter, self).add_figure(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        if can_log():
            super(DistSummaryWriter, self).add_graph(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if can_log():
            super(DistSummaryWriter, self).add_histogram(*args, **kwargs)
    
    def add_image(self, *args, **kwargs):
        if can_log():
            super(DistSummaryWriter, self).add_image(*args, **kwargs)

    def close(self):
        if can_log():
            super(DistSummaryWriter, self).close()


import tqdm


def dist_tqdm(obj, *args, **kwargs):
    if can_log():
        return tqdm.tqdm(obj, *args, **kwargs)
    else:
        return obj

