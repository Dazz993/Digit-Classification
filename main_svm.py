import yaml
import argparse
from utils.dataset import get_numpy_dataset
from utils.utils import ObjectDict
from utils.metrics import accuracy
from sklearn import svm
import joblib
import time
from time import perf_counter as t

parser = argparse.ArgumentParser(description='Digit Classification Using SVM')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to configuration (default: none)')

current_time = time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())

def sgd_svm(train_dataset, train_labels, test_dataset, test_labels, n_samples=10000):
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, verbose=3))
    clf.fit(train_dataset[:n_samples], train_labels[:n_samples])
    score = clf.score(test_dataset, test_labels)
    print(f"Score: {score}")

    joblib.dump(clf, f'states/svm/sgd_svm_{current_time}.pkl')

def svc(train_dataset, train_labels, test_dataset, test_labels, n_samples=10000):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    begin_time = t()
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=4, gamma='scale', verbose=True))
    clf.fit(train_dataset[:n_samples], train_labels[:n_samples])
    score = clf.score(test_dataset, test_labels)
    print(f"Score: {score}")
    print(f"Time elapsed: {(t() - begin_time) / 60} mins")

    joblib.dump(clf, f'states/svm/sgd_svm_{current_time}.pkl')

def thundersvm(train_dataset, train_labels, test_dataset, test_labels):
    from thundersvm import SVC

    clf = SVC(kernel='sigmoid', verbose=True, gpu_id=2)
    clf.fit(train_dataset, train_labels)
    score = clf.score(test_dataset, test_labels)
    print(f"Score: {score}")

if __name__ == '__main__':
    # parse arguments and load configs
    args = parser.parse_args()
    with open(args.cfg, 'r') as configure_file:
        cfg_dict = yaml.load(configure_file, Loader=yaml.FullLoader)
    cfg = ObjectDict(cfg_dict)
    print(cfg)

    # load dataset
    train_dataset, train_labels, test_dataset, test_labels = get_numpy_dataset(path='./data/', feature_extraction=cfg.get('feature_extraction', None),
                                          pixels_per_cell=(4, 4), cells_per_block=(2, 2))

    print(f"Successfully load data, train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")

    # sgd_svm(train_dataset, train_labels, test_dataset, test_labels)
    # thundersvm(train_dataset, train_labels, test_dataset, test_labels)
    svc(train_dataset, train_labels, test_dataset, test_labels)