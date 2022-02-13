import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from time import time
from enum import Enum
from pytorch_model_summary import summary
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


class Ansi(Enum):
    """ Set print colors. """
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    ORANGE = '\033[33m'
    BLUE = '\033[94m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class MLP(nn.Module):
    """ MLP module. """

    def __init__(
            self,
            input_dim,
            hidden_size=32,
            dropout=0,
            n_classes=2,
    ):
        super().__init__()
        self.dropout = dropout
        self.fc = nn.Linear(input_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dp = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        out = F.relu(self.fc(X))
        # out = self.bn(out)
        # if self.config['verbose']: print(out.shape)
        # out = self.dense(F.dropout(out, p=self.dropout))
        out = self.dense(self.dp(out))
        return out


class Classifier():
    """ Classifier (Pytorch version). """

    def __init__(
        self,
        module,
        module_args,
        optimizer,
        opt_args,
        train_args=None,
    ):
        self.module = module
        self.module_args = module_args
        self.optimizer = optimizer
        self.opt_args = opt_args
        self.config = {'batch_size': 8, 'max_epochs': 100, 'earlystop': 10, 'clip_norm': 5,
                       'verbose': 1, 'use_best': True, 'normalize': True, 'pca': False, 'n_components': None, }
        if train_args is not None:
            self.config.update(train_args)

    def get_model(self):
        """ Get model from `module` and `module_args`, and save the initial state. """
        self.model = self.module(**self.module_args)
        if not hasattr(self, 'init_state'):
            self.init_state = copy.deepcopy(self.model.state_dict())

    def get_optim(self):
        """ Get optimizer from `optimizer` and `opt_args`. """
        self.optim = self.optimizer(self.model.parameters(), **self.opt_args)

    def summary(self, x, **kwargs):
        """ Summary the architecture of the model. """
        # default: {*x, batch_size=-1, show_input=False, show_hierarchical=False,
        # print_summary=False, max_depth=1, show_parent_layers=False}
        x = torch.from_numpy(x.astype(np.float)).float() if isinstance(x, np.ndarray) else x
        kwargs.setdefault('print_summary', True)
        self.get_model()
        summary(self.model, x, **kwargs)

    def save(self, filepath):
        """ Save model. """
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        """ Load model. """
        return self.model.load_state_dict(torch.load(filepath))

    def initialize(self):
        """ Initialize module, optimizer (, callbacks, history). """
        if not hasattr(self, 'model'):
            self.get_model()
        else:
            self.model.load_state_dict(self.init_state)
        self.get_optim()

    def fit(self, X, y, verbose=None, shuffle=True, split=5):
        """ Train and validate model. """
        verbose = self.config['verbose'] if verbose is None else verbose
        batch_size = self.config['batch_size']
        if shuffle:
            pid = np.random.permutation(len(y))
            X, y = X[pid], y[pid]
        if split:
            # split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1/split, shuffle=True, random_state=0)
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        else:
            train_dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # get or initialize models and optimizers
        self.initialize()
        # Set lr scheduler, earlystopping and verbose
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim)
        earlystop_flag, lowest_loss, tol = 0, np.inf, 0.00
        best_logs = [np.inf, np.inf, np.inf]
        if verbose:
            print(("{:<7}"+"{:<10}"*4).format(
                'epoch', 'trn_loss', 'val_loss', 'val_acc', 'duration'))
            print(("{:<7}"+"{:<10}"*4).format(
                '-----', '--------', '--------', '--------', '--------'))
        # Train
        self.model.train()
        for epoch in range(self.config['max_epochs']):
            trn_loss, tst_loss, tst_acc, ts = 0, 0, [], time()
            for _, batch in enumerate(train_dataloader):
                self.optim.zero_grad()
                trn_x, trn_y = batch
                pred_y = self.model.forward(trn_x)
                loss = F.cross_entropy(pred_y, trn_y)  # Size: [32, 2], [32]
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_norm'])
                self.optim.step()
                trn_loss += loss.item()

            if split:
                # Validation
                with torch.no_grad():
                    # if test data is non-empty, ignore warning
                    for _, batch in enumerate(test_dataloader):
                        tst_x, tst_y = batch
                        pred_y = self.model.forward(tst_x)
                        loss = F.cross_entropy(pred_y, tst_y)
                        tst_loss += loss.item()
                        tst_acc.append(accuracy_score(
                            tst_y, np.argmax(pred_y.cpu().data.numpy(), axis=-1)))
                        
            else:
                tst_loss = trn_loss
                tst_acc.append(-1)
            # print logs
            if verbose and (epoch % verbose == 0):
                best_logs = self.print_logs(
                    best_logs, [epoch+1, trn_loss, tst_loss, np.array(tst_acc).mean(), time()-ts])
            # scheduler
            scheduler.step(tst_loss)
            # earlystopping
            if tst_loss < lowest_loss + tol:
                earlystop_flag = epoch
                lowest_loss = tst_loss
                self.bestmodel = copy.deepcopy(self.model.state_dict())
            if epoch - earlystop_flag >= self.config['earlystop']:
                if verbose:
                    print('Early stopping.')
                break

    def test(self, X, y):
        """ Test and evaluate. """
        scores = {}
        if self.config['use_best']:
            self.model.load_state_dict(self.bestmodel)
        with torch.no_grad():
            X = torch.FloatTensor(X.astype(np.float64))
            preds = self.model.forward(X)
            logits = F.softmax(preds).cpu().data.numpy()
            preds = np.argmax(logits, axis=-1)
        scores['acc'] = accuracy_score(y, preds)
        scores['prec'] = precision_score(y, preds)
        scores['rec'] = recall_score(y, preds)
        scores['f1'] = f1_score(y, preds)
        scores['fprs'], scores['tprs'], _ = roc_curve(y, logits[:, 1])
        scores['auc'] = auc(scores['fprs'], scores['tprs'])  # roc_auc_score(y, logits[:, 1])
        return scores

    def fit_test(self, X_trn, y_trn, X_tst, y_tst, times=1, verbose=0, split=5, toarray=True):
        """ Train and test data with several times. """
        scores = {}
        if self.config['normalize']:
            scaler = StandardScaler()
            X_trn = scaler.fit_transform(X_trn)
            X_tst = scaler.transform(X_tst)
        if self.config['pca']:
            ppca = PCA(self.config['n_components'])
            X_trn = ppca.fit_transform(X_trn)
            X_tst = ppca.transform(X_tst)
        for _ in range(times):
            self.fit(X_trn, y_trn, verbose=verbose, split=split)
            tst_scores = self.test(X_tst, y_tst)
            for key, val in tst_scores.items():
                if key not in scores.keys():
                    scores[key] = []
                scores[key].append(val)
        if toarray:
            for key, val in scores.items():
                scores[key] = np.asarray(val)
        return scores

    def cross_validate(self, X, y, cv=5, times=None, verbose=0, split=0):
        """ k times k-fold cross-validation. """
        scores = {}
        times = 1 if times is None else times
        for _ in range(times):
            pid = np.random.permutation(len(y))
            X, y = X[pid], y[pid]
            cv = len(y) if cv == -1 else cv
            kf = KFold(n_splits=cv, shuffle=True, random_state=0)
            for trn_idx, tst_idx in kf.split(X):
                tst_scores = self.fit_test(
                    X[trn_idx], y[trn_idx], X[tst_idx], y[tst_idx], times=1, verbose=verbose, split=split, toarray=False)
                for key, val in tst_scores.items():
                    if key not in scores.keys():
                        scores[key] = []
                    scores[key].append(val)
        for key, val in scores.items():
            scores[key] = np.asarray(val)
        return scores

    def cv_test(self, features, labels, tst_features, test_ad, cv=5, times=5, split=5):
        """ Cross validation and test. """
        scores = self.cross_validate(features, labels, cv=cv, times=times, split=split)
        print("cv:")
        self.print_scores(scores)
        tst_scores = self.fit_test(features, labels, tst_features, test_ad, times=times, split=split)
        print("tst:")
        self.print_scores(tst_scores)
        return scores, tst_scores

    def print_logs(self, best_logs, ep_logs):
        """ Pretty print logs while training. """
        values = [ep_logs[1], ep_logs[2], -ep_logs[3]]
        colors = [color.value for color in Ansi if color != color.ENDC]
        logs = ['{:<7}'.format(ep_logs[0]),
                '{:<10}'.format('{:.4f}'.format(values[0])),
                '{:<10}'.format('{:.4f}'.format(values[1])),
                '{:<10}'.format('{:.4f}'.format(-values[2])),
                '{:<10}'.format('{:.4f}'.format(ep_logs[-1]))]
        for i in range(len(best_logs)):
            if values[i] <= best_logs[i]:
                best_logs[i] = values[i]
                logs[i+1] = colors[i] + logs[i+1] + Ansi.ENDC.value
        print(''.join(logs))
        return best_logs

    def print_scores(self, scores):
        """ Pretty print scores. """
        keys = []
        values = []
        for key, val in scores.items():
            try:
                values.append("{:<12}".format("{:.2f}({:.2f})".format(val.mean(), val.std())))
                keys.append("{:<12}".format(key))
            except:
                continue
        print(''.join(keys) + '\n' + ''.join(values))
