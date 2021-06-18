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

def analyse_parameters(model):
    weight_list = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_list.append(param)

    for i, weight in enumerate(weight_list):
        print("=" * 10, f" weight id {i} ", "=" * 10)
        print(f"max: {torch.max(weight)}, min: {torch.min(weight)}")
        print(f"mean: {torch.mean(weight)}")
        print(f"std: {torch.std(weight)}")
        print(f"number of parameters below 1e-3: {torch.sum(torch.abs(weight) < 0.001)}")
        print(f"size: {weight.shape}")

    return weight_list

if __name__ == '__main__':
    # parse arguments and load configs
    args = parser.parse_args()
    with open(args.cfg, 'r') as configure_file:
        cfg_dict = yaml.load(configure_file, Loader=yaml.FullLoader)
    cfg = ObjectDict(cfg_dict)
    print(cfg)

    # define some parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dicts = torch.load('states/LogisticRegression_TenBinaryClassifier_WithRidge_HOG_2021_06_17_12_42_54/checkpoint_LogisticRegression_TenBinaryClassifier_WithRidge_HOG_best.tar')

    # load dataset and dataloader
    _, test_dataset = get_dataset(path='./data/', feature_extraction=cfg.get('feature_extraction', None), pixels_per_cell=(2, 2), cells_per_block=(3, 3))
    # train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
    #                                                num_workers=cfg.num_workers, pin_memory=True)
    # test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False,
    #                                                num_workers=cfg.num_workers, pin_memory=True)

    # define LogsiticRegreesion model
    input_size = test_dataset[0][0].shape
    models = []
    for i in range(10):
        model = LinearModel(input_size=input_size, num_classes=1).to(device)
        model.load_state_dict(state_dicts['state_dicts'][i])
        models.append(model)

    weight_list = analyse_parameters(models[0])