import os
import yaml
import time
import torch
import argparse
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm
from time import perf_counter as t
from models import VAE, DFCVAE
from utils.dataset import get_dataset, get_VAE_dataset
from utils.utils import ObjectDict, AverageMeter, save_checkpoint
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Pytorch Implementation')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to configuration (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    help='the id of gpu to use (default: 0)')

def train_one_epoch(model, optimizer, epoch, train_dataloader, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = t()

    for batch_idx, (input, _) in enumerate(train_dataloader):
        # process data loading time
        data_time.update(t() - end)

        # model execution
        input = input.to(device)
        output = model(input)

        loss, reconstruction_loss, kl_loss = model.loss(*output, kl_weight = cfg.batch_size / cfg.n_train_samples)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # process output and other measurements
        loss = loss.float()
        losses.update(loss, input.size(0))

        # process batch time
        batch_time.update(t() - end)
        end = t()

        if batch_idx % cfg.print_frequency == 0:
            print(f'Epoch: [{epoch}]/[{cfg.epochs}],\t'
                  f'Batch: [{batch_idx}]/[{len(train_dataloader)}],\t'
                  f'Time: {batch_time.val:.4f} ({batch_time.avg:.4f}),\t'
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f})')

    return losses.avg

@torch.no_grad()
def validate(model, epoch, test_dataloader, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = t()

    sample_images(model, epoch=epoch, test_dataloader=test_dataloader, device=cfg.device, cfg=cfg)

    for batch_idx, (input, target) in enumerate(test_dataloader):
        # process data loading time
        data_time.update(t() - end)

        # model execution
        input = input.to(device)
        output = model(input)

        loss, reconstruction_loss, kl_loss = model.loss(*output, kl_weight=cfg.batch_size / cfg.n_test_samples)

        # process output and other measurements
        loss = loss.float()
        losses.update(loss, input.size(0))

        # process batch time
        batch_time.update(t() - end)
        end = t()

        # print(f'Epoch: [{epoch}]/[{cfg.epochs}]\t'
        #       f'Batch: [{batch_idx}]/[{len(val_dataloader)}]\t'
        #       f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        #     #   f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
        #       f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #       f'f1_score {f1_micro.val:.4f} {f1_macro.val:.4f}')

    print(f'* Epoch: {epoch}, test_loss: {losses.avg:.4f}\n')

    return losses.avg

def sample_images(model, epoch, test_dataloader, device, cfg):
    '''Refer to: https://github.com/AntixK/PyTorch-VAE/blob/master/experiment.py
    '''
    # Get sample reconstruction image
    test_input, test_label = next(iter(test_dataloader))
    test_input = test_input.to(device)
    test_label = test_label.to(device)
    recons = model.generate(test_input)

    root = f'docs/vae_figs/{cfg.pid}'
    if not os.path.exists(root):
        os.makedirs(root)
    vutils.save_image(recons.data, f"{root}/recons_{epoch}.png", normalize=True, nrow=12)

    samples = model.sample(144, device)
    vutils.save_image(samples.cpu().data, f"{root}/{epoch}.png", normalize=True, nrow=12)

    del test_input, recons  # , samples

def train(model, optimizer, lr_scheduler, train_dataloader, test_dataloader, writer, cfg):
    best_test_acc, best_test_loss, best_epoch = 0, float('inf'), 0
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model=model, optimizer=optimizer, epoch=epoch, train_dataloader=train_dataloader, cfg=cfg)
        test_loss = validate(model=model, epoch=epoch, test_dataloader=test_dataloader, cfg=cfg)
        lr_scheduler.step()

        save_dict = {
            'epoch': epoch + 1,
            'loss': test_loss,
            'state_dict': model.state_dict()
        }

        is_best = False
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            is_best = True
        save_checkpoint(epoch, save_dict, is_best, cfg, path=cfg.checkpoint_path)

        writer.add_scalar(f'loss/train_loss', train_loss, epoch)
        writer.add_scalar(f'loss/test_loss', test_loss, epoch)

    print(f"Best test loss: {best_test_loss:.8f}, best epoch: {best_epoch}")


if __name__ == '__main__':
    # parse arguments and load configs
    args = parser.parse_args()
    with open(args.cfg, 'r') as configure_file:
        cfg_dict = yaml.load(configure_file, Loader=yaml.FullLoader)
    cfg = ObjectDict(cfg_dict)
    cfg['gpu'] = args.gpu
    print(cfg)

    # define some parameters
    cfg['device'] = device = f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu'
    cfg['pid'] = pid = f"{cfg.network}{cfg.get('layers', '')}_{cfg.learning_rate}_{time.strftime('_%Y_%m_%d_%H_%M_%S', time.localtime())}"
    cfg['checkpoint_path'] = f'./states/{pid}'

    # define summary write
    writer = SummaryWriter(f'./docs/tensorboard_logs/{pid}')

    # load dataset and dataloader
    train_dataset, test_dataset = get_dataset(path='./data/')
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                   num_workers=cfg.num_workers, pin_memory=True)
    cfg['n_train_samples'] = len(train_dataset)
    cfg['n_test_samples'] = len(test_dataset)

    # define LogsiticRegreesion model
    if cfg.network == 'VAE':
        model = VAE(inplanes=3, input_size=32, latent_dim=128, hidden_dims=(32, 64, 128, 256, 512)).to(device)
    elif cfg.network == 'DFCVAE':
        model = DFCVAE(inplanes=3, input_size=32, latent_dim=128, hidden_dims=(32, 64, 128, 256, 512)).to(device)
    else:
        raise NotImplementedError
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    # define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.gamma)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, train_dataloader=train_dataloader, test_dataloader=test_dataloader, writer=writer, cfg=cfg)