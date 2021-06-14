import os
import torch

class ObjectDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(epoch, save_dict, is_best, cfg, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(save_dict, os.path.join(path, f'checkpoint_{cfg.network}_{epoch}.tar'))
    if is_best:
        torch.save(save_dict, os.path.join(path, f'checkpoint_{cfg.network}_best.tar'))