import yaml
import time
import torch
import argparse
import torch.utils.data
from tqdm import tqdm
from time import perf_counter as t
from models import LogsiticRegression
from utils.dataset import get_dataset
from utils.metrics import accuracy
from utils.utils import ObjectDict, AverageMeter, save_checkpoint

parser = argparse.ArgumentParser(description='Pytorch CelebA Implementation')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to configuration (default: none)')

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
        target = target.to(device).float().requires_grad_()
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
                  f'Acc: {acc.val:.4f} ({acc.avg:.4f})')

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
        target = target.to(device).float()
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

    print(f'\n * Epoch: {epoch}, Acc: {accs.avg:.4f}, test_loss: {losses.avg:.4f}')

    return losses.avg, accs.avg

def train(model, criterion, optimizer, lr_scheduler, train_dataloader, test_dataloader, cfg):
    best_test_acc, best_val_loss = 0, float('inf')
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
            best_val_loss = test_loss
            is_best = True
        save_checkpoint(epoch, save_dict, is_best, cfg, path=f'./states/{cfg.network + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())}')

if __name__ == '__main__':
    # parse arguments and load configs
    args = parser.parse_args()
    with open(args.cfg, 'r') as configure_file:
        cfg_dict = yaml.load(configure_file, Loader=yaml.FullLoader)
    cfg = ObjectDict(cfg_dict)
    print(cfg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset and dataloader
    train_dataset, test_dataset = get_dataset(path='./data/', preprocess=None)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                   num_workers=cfg.num_workers, pin_memory=True)

    # define LogsiticRegreesion model
    model = LogsiticRegression(input_size=(3, 32, 32), num_classes=10).to(device)
    # define criterion
    criterion = torch.nn.CrossEntropyLoss
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    # define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    train(model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, train_dataloader=train_dataloader, test_dataloader=test_dataloader, cfg=cfg)