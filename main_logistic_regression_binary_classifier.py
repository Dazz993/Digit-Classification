import yaml
import time
import torch
import argparse
import torch.utils.data
from tqdm import tqdm
from time import perf_counter as t
from torch.utils.tensorboard import SummaryWriter
from models import LinearModel
from utils.dataset import get_dataset
from utils.metrics import binary_classifier_accuracy, accuracy
from utils.utils import ObjectDict, AverageMeter, save_checkpoint
from utils.losses import logistic_regression_loss, regularization_loss

parser = argparse.ArgumentParser(description='Pytorch Digit Classification Implementation')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to configuration (default: none)')

def train_one_epoch(models, criterion, optimizers, epoch, train_dataloader, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    train_losses_per_model = [AverageMeter() for _ in range(10)]
    train_accs_per_model = [AverageMeter() for _ in range(10)]

    for batch_idx, (input, target) in enumerate(train_dataloader):
        input = input.to(device)

        for i in range(10):
            model = models[i]
            optimizer = optimizers[i]

            model.train()

            end = t()

            # process data loading time
            data_time.update(t() - end)

            # model execution
            _target = torch.where(target == i, 1, 0).to(device)
            _target = _target.reshape(-1, 1).float()

            output = model(input)

            loss = criterion(output, _target)
            if cfg.regularization != -1:
                loss += regularization_loss(model, lam=cfg.lam, p=cfg.regularization)


            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # process output and other measurements
            loss = loss.float()
            # output = output.detach().cpu().numpy()
            # target = target.detach().cpu().numpy()
            acc = binary_classifier_accuracy(output=output, y_true=_target)

            losses.update(loss, input.size(0))
            accs.update(acc, input.size(0))

            train_losses_per_model[i].update(loss, input.size(0))
            train_accs_per_model[i].update(loss, input.size(0))

            # process batch time
            batch_time.update(t() - end)

        if batch_idx % cfg.print_frequency == 0:
            print(f'Epoch: [{epoch}]/[{cfg.epochs}],\t'
                  f'Batch: [{batch_idx}]/[{len(train_dataloader)}],\t'
                  f'Time: {batch_time.val:.4f} ({batch_time.avg:.4f}),\t'
                  #   f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}),\t'
                  f'Acc: {accs.val:.4f} ({accs.avg:.4f})')

    return [train_losses_per_model[i].avg for i in range(10)], [train_accs_per_model[i].avg for i in range(10)], [models[i].state_dict() for i in range(10)]

@torch.no_grad()
def validate(models, criterion, epoch, test_dataloader, cfg):
    losses = [AverageMeter() for _ in range(10)]
    accs = [AverageMeter() for _ in range(10)]

    for batch_idx, (input, target) in enumerate(test_dataloader):

        for i in range(10):
            model = models[i]
            model.eval()

            # model execution
            input = input.to(device)
            _target = torch.where(target == i, 1, 0).to(device)
            _target = _target.reshape(-1, 1).float()

            output = model(input)
            loss = criterion(output, _target)

            # process output and other measurements
            loss = loss.float()
            # output = output.detach().cpu().numpy()
            # target = target.detach().cpu().numpy()
            acc = binary_classifier_accuracy(output=output, y_true=_target)
            losses[i].update(loss, input.size(0))
            accs[i].update(acc, input.size(0))

            # print(f'Epoch: [{epoch}]/[{cfg.epochs}]\t'
            #       f'Batch: [{batch_idx}]/[{len(val_dataloader)}]\t'
            #       f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')

    average_accuracy = sum([accs[i].avg for i in range(10)]) / 10
    average_losses = sum([losses[i].avg for i in range(10)]) / 10

    print(f'* Epoch: {epoch}, Acc: {average_accuracy:.4f}, test_loss: {average_losses:.4f}\n')

    return [losses[i].avg for i in range(10)], [accs[i].avg for i in range(10)]

def train(models, criterion, optimizers, train_dataloader, test_dataloader, writer, cfg):
    best_test_acc = [0 for _ in range(10)]
    best_test_loss = [float('inf') for _ in range(10)]
    best_state_dicts = [models[i].state_dict() for i in range(10)]
    best_acc_for_all = 0
    for epoch in range(cfg.epochs):
        train_losses, train_accs, model_state_dicts = train_one_epoch(models=models, criterion=criterion, optimizers=optimizers, epoch=epoch, train_dataloader=train_dataloader, cfg=cfg)
        test_losses, test_acces = validate(models=models, criterion=criterion, epoch=epoch, test_dataloader=test_dataloader, cfg=cfg)

        for i in range(10):
            if test_losses[i] < best_test_loss[i]:
                best_test_acc[i] = test_acces[i]
                best_test_loss[i] = test_losses[i]
                best_state_dicts[i] = model_state_dicts[i]

        save_dict = {
            'epoch': epoch + 1,
            'loss': best_test_loss,
            'score': best_test_acc,
            'state_dicts': best_state_dicts
        }

        for i in range(10):
            writer.add_scalar(f'loss/train_loss/model_{i}', train_losses[i], epoch)
            writer.add_scalar(f'loss/test_loss/model_{i}', test_losses[i], epoch)

        print("Test for test dataset")
        is_best = False
        acc_for_all = test_for_all(models=models, model_state_dicts=best_state_dicts, test_dataloader=test_dataloader, cfg=cfg)
        if acc_for_all > best_acc_for_all:
            best_acc_for_all = acc_for_all
            is_best = True

        save_checkpoint(epoch, save_dict, is_best, cfg, path=cfg.checkpoint_path)

    return best_test_acc, best_test_loss, best_state_dicts, best_acc_for_all

@torch.no_grad()
def test_for_all(models, model_state_dicts, test_dataloader, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    for batch_idx, (input, target) in enumerate(test_dataloader):
        input = input.to(device)

        output_metrics = torch.zeros(input.shape[0], 10)

        for i in range(10):
            model = models[i]
            model.load_state_dict(model_state_dicts[i])
            model.eval()

            output = model(input).reshape(-1)

            output_metrics[:, i] = output.detach().cpu()

        acc = accuracy(output=output_metrics, y_true=target)
        accs.update(acc, input.size(0))

    print(f'* Acc: {accs.avg:.4f}\n')

    return accs.avg


if __name__ == '__main__':
    # parse arguments and load configs
    args = parser.parse_args()
    with open(args.cfg, 'r') as configure_file:
        cfg_dict = yaml.load(configure_file, Loader=yaml.FullLoader)
    cfg = ObjectDict(cfg_dict)
    print(cfg)

    # define some parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pid = cfg.network + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())
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
    input_size = train_dataset[0][0].shape
    models = []
    for i in range(10):
        model = LinearModel(input_size=input_size, num_classes=1).to(device)
        models.append(model)

    # define criterion
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = logistic_regression_loss
    # define optimizer
    optimizers = []
    for i in range(10):
        optimizer = torch.optim.AdamW(models[i].parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        optimizers.append(optimizer)

    best_test_acc, best_test_loss, best_state_dicts, best_acc_for_all = train(models=models, criterion=criterion, optimizers=optimizers, train_dataloader=train_dataloader, test_dataloader=test_dataloader, writer=writer, cfg=cfg)

    print(f"Best for all: {best_acc_for_all}")

    # test_for_all(models=models, model_state_dicts=best_state_dicts, test_dataloader=test_dataloader, cfg=cfg)