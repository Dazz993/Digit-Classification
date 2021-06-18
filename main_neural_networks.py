import yaml
import time
import torch
import argparse
import torch.utils.data
from tqdm import tqdm
from time import perf_counter as t
from models import LinearModel, SimpleConv, vgg, resnet
from utils.dataset import get_dataset
from utils.metrics import accuracy
from utils.utils import ObjectDict, AverageMeter, save_checkpoint
from utils.losses import regularization_loss
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Pytorch Implementation')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to configuration (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    help='the id of gpu to use (default: 0)')

def train_one_epoch(model, criterion, optimizer, epoch, train_dataloader, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    model.train()

    end = t()

    for batch_idx, (input, target) in enumerate(train_dataloader):
        # process data loading time
        data_time.update(t() - end)

        # model execution
        input = input.to(device)
        target = target.to(device)
        output = model(input)

        loss = criterion(output, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # process output and other measurements
        loss = loss.float()

        # output = output.detach().cpu().numpy()
        # target = target.detach().cpu().numpy()
        acc = accuracy(output=output, y_true=target)
        losses.update(loss, input.size(0))
        accs.update(acc, input.size(0))

        # process batch time
        batch_time.update(t() - end)
        end = t()

        if batch_idx % cfg.print_frequency == 0:
            print(f'Epoch: [{epoch}]/[{cfg.epochs}],\t'
                  f'Batch: [{batch_idx}]/[{len(train_dataloader)}],\t'
                  f'Time: {batch_time.val:.4f} ({batch_time.avg:.4f}),\t'
                  #   f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}),\t'
                  f'Acc: {accs.val:.4f} ({accs.avg:.4f})')

    return losses.avg, accs.avg

@torch.no_grad()
def validate(model, criterion, epoch, test_dataloader, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    model.eval()

    end = t()

    for batch_idx, (input, target) in enumerate(test_dataloader):
        # process data loading time
        data_time.update(t() - end)

        # model execution
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)

        # process output and other measurements
        loss = loss.float()
        # output = output.detach().cpu().numpy()
        # target = target.detach().cpu().numpy()
        acc = accuracy(output=output, y_true=target)
        losses.update(loss, input.size(0))
        accs.update(acc, input.size(0))

        # process batch time
        batch_time.update(t() - end)
        end = t()

        # print(f'Epoch: [{epoch}]/[{cfg.epochs}]\t'
        #       f'Batch: [{batch_idx}]/[{len(val_dataloader)}]\t'
        #       f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        #     #   f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
        #       f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #       f'f1_score {f1_micro.val:.4f} {f1_macro.val:.4f}')

    print(f'* Epoch: {epoch}, Acc: {accs.avg:.4f}, test_loss: {losses.avg:.4f}\n')

    return losses.avg, accs.avg

def train(model, criterion, optimizer, lr_scheduler, train_dataloader, test_dataloader, writer, cfg):
    best_test_acc, best_test_loss, best_epoch = 0, float('inf'), 0
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, train_dataloader=train_dataloader, cfg=cfg)
        test_loss, test_acc = validate(model=model, criterion=criterion, epoch=epoch, test_dataloader=test_dataloader, cfg=cfg)
        lr_scheduler.step()

        if cfg.multigpu:
            save_dict = {
                'epoch': epoch + 1,
                'loss': test_loss,
                'score': test_acc,
                'state_dict': model.module.state_dict()
            }
        else:
            save_dict = {
                'epoch': epoch + 1,
                'loss': test_loss,
                'score': test_acc,
                'state_dict': model.state_dict()
            }

        is_best = False
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_epoch = epoch
            is_best = True
        save_checkpoint(epoch, save_dict, is_best, cfg, path=cfg.checkpoint_path)

        writer.add_scalar(f'loss/train_loss', train_loss, epoch)
        writer.add_scalar(f'loss/test_loss', test_loss, epoch)
        writer.add_scalar(f'acc/train_acc', train_acc, epoch)
        writer.add_scalar(f'acc/test_acc', test_acc, epoch)

    print(f"Best test loss: {best_test_loss:.8f}, best test accuracy: {best_test_acc:.8f}, best epoch: {best_epoch}")


if __name__ == '__main__':
    # parse arguments and load configs
    args = parser.parse_args()
    with open(args.cfg, 'r') as configure_file:
        cfg_dict = yaml.load(configure_file, Loader=yaml.FullLoader)
    cfg = ObjectDict(cfg_dict)
    cfg['gpu'] = args.gpu
    print(cfg)

    # define some parameters
    device = f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu'
    pid = f"{cfg.network}{cfg.get('layers', '')}_{cfg.learning_rate}_{time.strftime('_%Y_%m_%d_%H_%M_%S', time.localtime())}"
    cfg['checkpoint_path'] = f'./states/{pid}'

    # define summary write
    writer = SummaryWriter(f'./docs/tensorboard_logs/{pid}')

    # load dataset and dataloader
    train_dataset, test_dataset = get_dataset(path='./data/', feature_extraction=cfg.get('feature_extraction', None), pixels_per_cell=(2, 2), cells_per_block=(3, 3))
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                   num_workers=cfg.num_workers, pin_memory=True)

    # define LogsiticRegreesion model
    if cfg.network == 'SimpleConv':
        model = SimpleConv(num_classes=10).to(device)
    elif cfg.network == 'VGG':
        model = vgg(layers=cfg.layers, num_classes=10).to(device)
    elif cfg.network == 'ResNet':
        model = resnet(layers=cfg.layers, num_classes=10).to(device)
    else:
        raise NotImplementedError
    # define criterion
    criterion = torch.nn.CrossEntropyLoss()
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    # define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train(model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, train_dataloader=train_dataloader, test_dataloader=test_dataloader, writer=writer, cfg=cfg)