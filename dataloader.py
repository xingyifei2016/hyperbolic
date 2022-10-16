import torch
import numpy as np
from skimage import color
import torchvision
import logging
import os
import random
from pdb import set_trace as st

LOG = logging.getLogger('base')

os.environ['PYTHONHASHSEED'] = str(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def convert_real_imag(inputs):
    # Converts a [B, C, H, W] input into a [B, 2, C, H, W] complex representation
    real = np.real(inputs)
    imag = np.imag(inputs)
    return np.stack([real, imag], axis=1)



def generate_CIFAR_dataloader(split_percent=10, data_path='./CIFAR/', train_batch=256,
                              test_batch=256, val_batch=256, seed=0, remove_train=0, 
                              dset_type='cifar10', *args, **kwargs):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    assert dset_type in ['cifar10', 'cifar100', 'svhn']
    if dset_type == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True)
        test = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True)
        y_train = np.array(train_set.targets)
        y_test = np.array(test.targets)
        x_train = train_set.data
        x_test = test.data
    elif dset_type == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True)
        test = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True)
        y_train = np.array(train_set.targets)
        y_test = np.array(test.targets)
        x_train = train_set.data
        x_test = test.data
    else:
        train_set = torchvision.datasets.SVHN(
            root=data_path, split='train', download=True)
        test = torchvision.datasets.SVHN(
            root=data_path, split='test', download=True)
        y_train = np.array(train_set.labels)
        y_test = np.array(test.labels)
        x_train = train_set.data.transpose(0, 2, 3, 1)
        x_test = test.data.transpose(0, 2, 3, 1)

    x_train = x_train.astype(float)/255.0
    x_test = x_test.astype(float)/255.0

    
    train_idx = np.random.permutation(np.arange(x_train.shape[0]))
    x_train = x_train[train_idx].transpose(0, 3, 1, 2)
    y_train = y_train[train_idx]
    x_test = x_test.transpose(0, 3, 1, 2)
    
    
    if remove_train != 0:
        total_size = len(x_train)
        print("Original train set size: " + str(total_size))
        print("Using this much data: "+str(int(total_size*remove_train)))
        x_train = x_train[:int(total_size*remove_train)]
        y_train = y_train[:int(total_size*remove_train)]
    
    
    total_size = len(x_train)
    trainval_subset = int(split_percent*total_size/100)
    train_size = int(trainval_subset*0.9)
    val_size = int(trainval_subset*0.1)

    x_train, x_val = x_train[:train_size], x_train[train_size:train_size+val_size]
    y_train, y_val = y_train[:train_size], y_train[train_size:train_size+val_size]
    LOG.info("Using {} Train, {} Validation, and {} Test examples".format(
        len(x_train), len(x_val), len(x_test)))

    

    data_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).type(
        torch.float32), torch.from_numpy(y_train).type(torch.LongTensor))
    data_val = torch.utils.data.TensorDataset(torch.from_numpy(x_val).type(
        torch.float32), torch.from_numpy(y_val).type(torch.LongTensor))
    data_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).type(
        torch.float32), torch.from_numpy(y_test).type(torch.LongTensor))

    params_train = {'batch_size': train_batch,
                    'shuffle': True,
                    'worker_init_fn': worker_init_fn}
    params_val = {'batch_size': val_batch,
                  'shuffle': False,
                  'worker_init_fn': worker_init_fn}
    params_test = {'batch_size': test_batch,
                   'shuffle': False,
                   'worker_init_fn': worker_init_fn}

    train_generator = torch.utils.data.DataLoader(
        dataset=data_train, pin_memory=True, **params_train)
    val_generator = torch.utils.data.DataLoader(
        dataset=data_val, pin_memory=True, **params_val)
    test_generator = torch.utils.data.DataLoader(
        dataset=data_test, pin_memory=True, **params_test)

    return train_generator, val_generator, test_generator