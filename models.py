import copy
import numpy as np
import regex as re
import pandas as pd
from enum import Enum
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from pytorch_model_summary import summary
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_curve, auc, roc_auc_score
import colorama
colorama.deinit()
colorama.init(strip=False)

# __all__ = ['Ansi', 'Baseline', 'Classifier', ...]


class Ansi(Enum):
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    ORANGE = '\033[33m'
    BLUE = '\033[94m'
    RED = '\033[31m'
    ENDC = '\033[0m'


def load_transformer_model_tokenizer(model_class, tokenizer_class, pretrained_weights):
    # ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return tokenizer, model


def bert_embed1d(text, tokenizer, model, max_len=512, trained=False):
    # Embedding function
    if not isinstance(text, pd.Series):
        text = pd.Series(text)
    tokenized = text.apply((lambda x: tokenizer.encode(
        x, add_special_tokens=True, max_length=max_len, truncation=True)))
    # pad so can be treated as one batch
    max_len = max([len(i) for i in tokenized.values])
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # attention mask - zero out attention scores where there is no input to be processed (i.e. is padding)
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    if trained:
        return input_ids, attention_mask
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # check if multiple GPUs are available
    if torch.cuda.is_available():
        model = model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    last_hidden_states = last_hidden_states[0]
    if device.type == 'cuda':
        last_hidden_states = last_hidden_states.cpu()
    features = last_hidden_states[:, 0, :].numpy()
    return features, attention_mask


class Baseline():
    def __init__(
        self,
        mode='lda',
        pre='pca',
        C=1,
        kernel='rbf',
        n_components=None,
    ):
        self.mode = mode
        self.pre = pre
        self.C = C
        self.kernel = kernel
        self.n_components = n_components

    def fit(self, X, y, shuffle=True):
        if shuffle:
            pid = np.random.permutation(len(y))
            X, y = X[pid], y[pid]
        pipe = []
        if self.pre == 'tfidf':
            # pipe.append(('scaler', StandardScaler()))
            pipe.append(('vec', TfidfVectorizer()))
            pipe.append(('func', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)))
            pipe.append(('scaler', StandardScaler()))
            pipe.append(('pca', PCA(n_components=self.n_components)))
        elif self.pre == 'sc':
            pipe.append(('scaler', StandardScaler()))
        elif self.pre == 'pca':
            pipe.append(('scaler', StandardScaler()))
            pipe.append(('pca', PCA(n_components=self.n_components)))
        if self.mode == 'svm':
            pipe.append(('cls', SVC(C=self.C, kernel=self.kernel, probability=True)))
        elif self.mode == 'lda':
            pipe.append(('cls', LDA()))
        elif self.mode == 'qda':
            pipe.append(('cls', QDA()))
        self.pipe = Pipeline(pipe)
        self.pipe.fit(X, y)

    def test(self, X, y):
        scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        preds = self.pipe.predict(X)
        logits = self.pipe.predict_proba(X)
        scores['acc'] = accuracy_score(y, preds)
        scores['prec'] = precision_score(y, preds)
        scores['rec'] = recall_score(y, preds)
        scores['f1'] = f1_score(y, preds)
        # scores['fprs'], scores['tprs'], _ = roc_curve(y, logits[:, 1])
        # scores['auc'] = auc(scores['fprs'], scores['tprs'])
        return scores

    def cross_validate(self, X, y, cv=5, times=None):
        scores = {}
        times = 1 if times is None else times
        for _ in range(times):
            pid = np.random.permutation(len(y))
            X, y = X[pid], y[pid]
            cv = len(y) if cv == -1 else cv
            kf = KFold(n_splits=cv, shuffle=True, random_state=0)
            for trn_idx, tst_idx in kf.split(X):
                self.fit(X[trn_idx], y[trn_idx])
                tst_scores = self.test(X[tst_idx], y[tst_idx])
                for key, val in tst_scores.items():
                    if key not in scores.keys():
                        scores[key] = []
                    scores[key].append(val)
        for key, val in scores.items():
            scores[key] = np.asarray(val)
        return scores

    def fit_test(self, X_trn, y_trn, X_tst, y_tst, times=25):
        scores = {}
        for _ in range(times):
            self.fit(X_trn, y_trn)
            tst_scores = self.test(X_tst, y_tst)
            for key, val in tst_scores.items():
                if key not in scores.keys():
                    scores[key] = []
                scores[key].append(val)
        for key, val in scores.items():
            scores[key] = np.asarray(val)
        return scores

    def cv_test(self, features, labels, tst_features, test_ad, cv=5, times=5, testtimes=1):
        scores = self.cross_validate(features, labels, cv=cv, times=times)
        print(". \tacc\t      prec\t    rec\t\t  f1")
        print("cv{}:".format(cv), self.scores_str(scores))
        # print("All accs:"+(("\n"+" {:.2f}"*cv)*times).format(*scores['acc']))
        tst_scores = self.fit_test(features, labels, tst_features, test_ad, times=testtimes)
        print("tst:", self.scores_str(tst_scores))
        return scores, tst_scores

    def scores_str(self, scores):
        return "{:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f})".format(
            scores['acc'].mean(), scores['acc'].std(), scores['prec'].mean(), scores['prec'].std(),
            scores['rec'].mean(), scores['rec'].std(), scores['f1'].mean(), scores['f1'].std())

    def print_scores(self, scores, mode='test'):
        if mode=='test':
            print(("{:<6}"*4).format('acc','prec','rec','f1'))
            table = []
            for _, val in scores.items():
                table.append("{:<6}".format("{:.2f}".format(val)))
            print(''.join(table))
        else:
            print(("{:<12}"*4).format('acc','prec','rec','f1'))
            table = []
            for _, val in scores.items():
                table.append("{:<12}".format("{:.2f}({:.2f})".format(val.mean(), val.std())))
            print(''.join(table))


class Classifier():
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
        config = {'batch_size': 32, 'max_epochs': 1000, 'earlystop': 10,
                  'verbose': 1, 'normalize': True, 'pca': False, 'n_components': None,'expand_dim': 0,}
        if train_args is not None:
            config.update(train_args)
        self.batch_size = config['batch_size']
        self.max_epochs = config['max_epochs']
        self.earlystop = config['earlystop']
        self.verbose = config['verbose']
        self.normalize = config['normalize']
        self.pca = config['pca']
        self.n_components = config['n_components']
        self.expand_dim = config['expand_dim']

    def summary(self, x, **kwargs):
        # *x, batch_size=-1, show_input=False, show_hierarchical=False,
        # print_summary=False, max_depth=1, show_parent_layers=False
        if self.expand_dim:
            x = np.expand_dims(x, self.expand_dim)
        x = torch.from_numpy(x.astype(np.float)).float() if isinstance(x, np.ndarray) else x
        # kwargs.setdefault('show_input', True) # if False, show output shape
        kwargs.setdefault('print_summary', True)
        self.get_model()
        summary(self.model, x, **kwargs)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        return self.model.load_state_dict(torch.load(filepath))

    def initialize(self):
        # Initialize module, optimizer, callbacks, history
        self.model.load_state_dict(self.init_state)
        for weight in self.model.parameters():
            try:
                self.model.init.orthogonal_(weight)
            except:
                continue

    def get_model(self):
        self.model = self.module(**self.module_args)

    def get_optim(self):
        # return self.optimizer(self.model.parameters(), **self.opt_args)
        for key in list(self.opt_args):
            try:
                return self.optimizer(self.model.parameters(), **self.opt_args)
            except:
                self.opt_args.pop(key)

    def fit(self, X, y, cv=5, verbose=None, shuffle=True, split=False):
        verbose = self.verbose if verbose is None else verbose
        cv = 10 if cv == -1 else cv
        X_lens = [X[m].shape[-1] for m in range(len(X))] if isinstance(X, list) else [X.shape[0]]
        X = np.concatenate(X, -1) if isinstance(X, list) else X
        if self.expand_dim:
            X = np.expand_dims(X, self.expand_dim)
        if shuffle:
            pid = np.random.permutation(len(y))
            X, y = X[pid], y[pid]
        # split data
        if split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1/cv, shuffle=True, random_state=0)
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            train_dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # initialize models and optimizers
        if not hasattr(self, 'init_state'):
            self.get_model()
            self.init_state = copy.deepcopy(self.model.state_dict())
        self.initialize()
        optimizer = self.get_optim()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        self.model.train()
        earlystop_flag, lowest_loss, tol = 0, np.inf, 0.00
        best_logs = [np.inf, np.inf, np.inf]
        if verbose:
            print(("{:<7}"+"{:<10}"*4).format(
                'epoch','trn_loss','val_loss','val_acc','duration'))
            print(("{:<7}"+"{:<10}"*4).format(
                '-----','--------','--------','--------','--------'))
        # Train
        for epoch in range(self.max_epochs):
            trn_loss, tst_loss, tst_acc, ts = 0, 0, [], time()
            for _, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                trn_x, trn_y = batch
                if len(X_lens) > 1:
                    trn_x = [trn_x[:, :X_lens[0]], trn_x[:, X_lens[0]:]]
                pred_y = self.model.forward(trn_x)
                loss = F.cross_entropy(pred_y, trn_y) # Size: [32, 2], [32]
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)  # hyperparameter: 5.0
                optimizer.step()
                trn_loss += loss.item()

            if split:
                with torch.no_grad():
                    for _, batch in enumerate(test_dataloader):
                        tst_x, tst_y = batch
                        if len(X_lens) > 1:
                            tst_x = [tst_x[:, :X_lens[0]], tst_x[:, X_lens[0]:]]
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
            if epoch - earlystop_flag >= self.earlystop:
                if verbose:
                    print('Early stopping.')
                break

        self.optim = optimizer
        return self.bestmodel

    def test(self, X, y):
        scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        self.model.load_state_dict(self.bestmodel)
        if self.expand_dim:
            X = np.expand_dims(X, self.expand_dim)
        with torch.no_grad():
            X = [torch.FloatTensor(X[m].astype(np.float64)) for m in range(len(X))] if isinstance(X, list) else torch.FloatTensor(X.astype(np.float64))
            preds = self.model.forward(X)
            logits = F.softmax(preds, dim=1).cpu().data.numpy()
            preds = np.argmax(logits, axis=-1)
        scores['acc'] = accuracy_score(y, preds)
        scores['prec'] = precision_score(y, preds)
        scores['rec'] = recall_score(y, preds)
        scores['f1'] = f1_score(y, preds)
        # scores['fprs'], scores['tprs'], _ = roc_curve(y, logits[:, 1])
        # scores['auc'] = auc(scores['fprs'], scores['tprs'])  # roc_auc_score(y, logits[:, 1])
        return scores

    def fit_test(self, X_trn, y_trn, X_tst, y_tst, times=25, toarray=True):
        scores = {}
        if self.normalize:
            if isinstance(X_trn, list):
                scalers = {}
                for m in range(len(X_trn)):
                    scalers[m] = StandardScaler()
                    X_trn[m] = scalers[m].fit_transform(X_trn[m])
                    X_tst[m] = scalers[m].transform(X_tst[m])
            else:
                scaler = StandardScaler()
                X_trn = scaler.fit_transform(X_trn)
                X_tst = scaler.transform(X_tst)
        if self.pca:
            ppca = PCA(self.n_components)
            X_trn = ppca.fit_transform(X_trn)
            X_tst = ppca.transform(X_tst)
        for _ in range(times):
            self.fit(X_trn, y_trn, verbose=0)
            tst_scores = self.test(X_tst, y_tst)
            for key, val in tst_scores.items():
                if key not in scores.keys():
                    scores[key] = []
                scores[key].append(val)
        if toarray:
            for key, val in scores.items():
                scores[key] = np.asarray(val)
        return scores

    def cross_validate(self, X, y, cv=5, times=None, verbose=0, split=True):
        # k times k-fold cross-validation
        scores = {}
        times = 1 if times is None else times
        for _ in range(times):
            pid = np.random.permutation(len(y))
            X = [X[m][pid] for m in range(len(X))] if isinstance(X, list) else X[pid]
            y = y[pid]
            cv = len(y) if cv == -1 else cv
            kf = KFold(n_splits=cv, shuffle=True, random_state=0)
            kfX = kf.split(X[0]) if isinstance(X, list) else kf.split(X)
            for trn_idx, tst_idx in kfX:
                X_trn = [X[m][trn_idx] for m in range(len(X))] if isinstance(X, list) else X[trn_idx]
                X_tst = [X[m][tst_idx] for m in range(len(X))] if isinstance(X, list) else X[tst_idx]
                tst_scores = self.fit_test(X_trn, y[trn_idx],
                                           X_tst, y[tst_idx], times=1, toarray=False)
                for key, val in tst_scores.items():
                    if key not in scores.keys():
                        scores[key] = []
                    scores[key].append(val)
        for key, val in scores.items():
            scores[key] = np.asarray(val)
        return scores

    def cv_test(self, features, labels, tst_features, test_ad, cv=5, times=5, testtimes=1):
        scores = self.cross_validate(features, labels, cv=cv, times=times)
        print(". \tacc\t      prec\t    rec\t\t  f1")
        print("cv{}:".format(cv), self.scores_str(scores))
        # print("All accs:"+(("\n"+" {:.2f}"*cv)*times).format(*scores['acc']))
        tst_scores = self.fit_test(features, labels, tst_features, test_ad, times=testtimes)
        print("tst:", self.scores_str(tst_scores))
        return scores, tst_scores

    def scores_str(self, scores):
        return "{:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f})".format(
            scores['acc'].mean(), scores['acc'].std(), scores['prec'].mean(), scores['prec'].std(),
            scores['rec'].mean(), scores['rec'].std(), scores['f1'].mean(), scores['f1'].std())

    def print_scores(self, scores, mode='test'):
        if mode=='test':
            print(("{:<6}"*4).format('acc','prec','rec','f1'))
            table = []
            for _, val in scores.items():
                table.append("{:<6}".format("{:.2f}".format(val)))
            print(''.join(table))
        else:
            print(("{:<12}"*4).format('acc','prec','rec','f1'))
            table = []
            for _, val in scores.items():
                table.append("{:<12}".format("{:.2f}({:.2f})".format(val.mean(), val.std())))
            print(''.join(table))
            
    def print_logs(self, best_logs, ep_logs):
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
