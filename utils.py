import dataloader
import config
import argparse
import os
import sys
import tqdm
import torch
import numpy as np
import model
import random
import coloredlogs
import logging
from torch.utils.tensorboard import SummaryWriter



def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir):
        print("FOLDER EXISTS")
    os.system('rm -r '+ckpt_dir)
    os.makedirs(ckpt_dir+'/runs', exist_ok=True)
    os.makedirs(ckpt_dir+'/code', exist_ok=True)
    # Makes a copy of all local code files to the log directory, just in case there are any problems
    os.system('cp *.py '+ckpt_dir+'/code/')
    # Makes a copy of all config files
    os.system('cp -r config '+ckpt_dir+'/code/')


LOG = logging.getLogger('base')
coloredlogs.install(level='DEBUG', logger=LOG)


def setup_logger(LOG, checkpoint_dir, debug=True):
    '''set up logger'''

    formatter = logging.Formatter('%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    LOG.setLevel(logging.DEBUG)

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "run.log")
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    LOG.addHandler(fh)


class Model_Tracker:
    def __init__(self, path, LOG):
        self.path = path
        self.name = None
        self.best_acc = 0
        self.LOG = LOG

        if self.path[-1] != '/':
            self.path = self.path+'/'

    def remove_old_save(self):
        if self.name:
            self.LOG.info("Removing Old Model Checkpoint at " +
                          self.path+self.name)
            os.remove(os.path.join(self.path, self.name))

    def create_new_save(self, name, net):
        self.LOG.info("New Best Validation Accuracy: "+str(self.best_acc))
        self.remove_old_save()
        self.name = name
        self.LOG.info("Creating New Model Checkpoint at " +
                      os.path.join(self.path, self.name))
        torch.save(net, os.path.join(self.path, self.name))

    def update(self, new_acc):
        better = new_acc > self.best_acc
        if better:
            self.best_acc = new_acc
        return better


xent = torch.nn.CrossEntropyLoss()


def loss_fn(x, y):
    return xent(x, y), (torch.argmax(x, dim=-1) == y).type(torch.cuda.FloatTensor).sum()



def log_init():
    # Initializes a logger
    LOG = logging.getLogger('base')
    coloredlogs.install(level='DEBUG', logger=LOG)
    return LOG

def setup_seed(seed):
    # Set up seed in all environments
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    return

def test_model(test_loader, net, current_iter, tracker, writer, logger, save_this_iter, cuda=False):
    net.eval()
    logger.info("Testing....")
    test_losses = []
    test_acc = []
    with torch.no_grad():
        for batch in tqdm.tqdm(iter(test_loader), dynamic_ncols=True):
            x, y = batch
            if cuda:
                loss, acc = loss_fn(net(x.cuda()), y.cuda())
            else:
                loss, acc = loss_fn(net(x), y)
            test_losses.append(loss.item())
            test_acc.append(acc.item())
    mean_test_acc = np.sum(test_acc)/len(test_loader.dataset)
    mean_test_loss = np.sum(test_losses)/len(test_loader.dataset)

    writer.add_scalar('Test Loss', mean_test_loss, current_iter)
    writer.add_scalar('Test Acc', mean_test_acc, current_iter)
    logger.info("Finished Testing! Mean Loss: {}, Acc: {}.".format(
        mean_test_loss, mean_test_acc))
    logger.info("Back to Training.")

    if save_this_iter:
        tracker.create_new_save('val_'+str(round(tracker.best_acc, 4))+'_test_'+str(
            round(mean_test_acc, 4))+'_'+str(current_iter+1).zfill(8)+'.pth', net)


def val_model(val_loader, net, current_iter, tracker, writer, logger, cuda=False):
    net.eval()
    logger.info("Validating....")
    val_losses = []
    val_acc = []

    with torch.no_grad():
        for idx, batch in tqdm.tqdm(enumerate(val_loader), dynamic_ncols=True):
            x, y = batch
            if cuda:
                loss, acc = loss_fn(net(x.cuda()), y.cuda())
            else: 
                loss, acc = loss_fn(net(x), y)
            val_losses.append(loss.item())
            val_acc.append(acc.item())

    mean_val_acc = np.sum(val_acc)/len(val_loader.dataset)
    mean_val_loss = np.sum(val_losses)/len(val_loader.dataset)
    writer.add_scalar('Validation Loss', mean_val_loss, current_iter)
    writer.add_scalar('Validation Acc', mean_val_acc, current_iter)
    logger.info("Finished Validation! Mean Loss: {}, Acc: {}".format(
        mean_val_loss, mean_val_acc))
    return tracker.update(mean_val_acc)

