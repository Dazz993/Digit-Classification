import yaml
import time
import torch
import argparse
import torch.utils.data
from tqdm import tqdm
from time import perf_counter as t
from models import LinearModel
from utils.dataset import get_dataset
from utils.metrics import binary_classifier_accuracy, accuracy
from utils.utils import ObjectDict, AverageMeter, save_checkpoint

parser = argparse.ArgumentParser(description='Pytorch CelebA Implementation')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to configuration (default: none)')

def train_one_epoch(model_id, model, criterion, optimizer, epoch, train_dataloader, cfg):
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
        target = torch.where(target == model_id, 1, 0).to(device)
        target = target.reshape(-1, 1).float()

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
        acc = binary_classifier_accuracy(output=output, y_true=target)
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
def validate(model_id, model, criterion, epoch, test_dataloader, cfg):
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
        target = torch.where(target == model_id, 1, 0).to(device)
        target = target.reshape(-1, 1).float()
        output = model(input)
        loss = criterion(output, target)

        # process output and other measurements
        loss = loss.float()
        # output = output.detach().cpu().numpy()
        # target = target.detach().cpu().numpy()
        acc = binary_classifier_accuracy(output=output, y_true=target)
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

def train(model_id, model, criterion, optimizer, train_dataloader, test_dataloader, cfg):
    best_test_acc, best_test_loss = 0, float('inf')
    best_state_dict = model.state_dict()
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model_id=model_id, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, train_dataloader=train_dataloader, cfg=cfg)
        test_loss, test_acc = validate(model_id=model_id, model=model, criterion=criterion, epoch=epoch, test_dataloader=test_dataloader, cfg=cfg)

        # if cfg.multigpu:
        #     save_dict = {
        #         'epoch': epoch + 1,
        #         'loss': test_loss,
        #         'score': test_acc,
        #         'state_dict': model.module.state_dict()
        #     }
        # else:
        #     save_dict = {
        #         'epoch': epoch + 1,
        #         'loss': test_loss,
        #         'score': test_acc,
        #         'state_dict': model.state_dict()
        #     }

        is_best = False
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_state_dict = model.state_dict()
            is_best = True
        # save_checkpoint(epoch, save_dict, is_best, cfg, path=cfg.checkpoint_path)

    print(f"Model ID: {model_id}, Best test loss: {best_test_loss:.8f}, best test accuracy: {best_test_acc:.8f}\n")

    return best_state_dict

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
    cfg['checkpoint_path'] = f'./states/{cfg.network + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())}'

    # load dataset and dataloader
    train_dataset, test_dataset = get_dataset(path='./data/', feature_extraction=cfg.get('feature_extraction', None), pixels_per_cell=(2, 2), cells_per_block=(2, 2))
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                   num_workers=cfg.num_workers, pin_memory=True)

    # define LogsiticRegreesion model
    input_size = train_dataset[0][0].shape
    models = []
    for i in range(10):
        models.append(LinearModel(input_size=input_size, num_classes=1).to(device))

    # define criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    # define optimizer
    optimizers = []
    for i in range(10):
        optimizers.append(torch.optim.AdamW(models[i].parameters(), lr=cfg.learning_rate))

    model_state_dicts = []
    for i in range(10):
        state_dict = train(model_id=i, model=models[i], criterion=criterion, optimizer=optimizers[i], train_dataloader=train_dataloader, test_dataloader=test_dataloader, cfg=cfg)
        model_state_dicts.append(state_dict)

    test_for_all(models=models, model_state_dicts=model_state_dicts, test_dataloader=test_dataloader, cfg=cfg)