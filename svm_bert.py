import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Classifier():
    def __init__(
        self,
        mode='svm',
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
        if self.pre == 'sc':
            pipe.append(('scaler', StandardScaler()))
        elif self.pre == 'pca':
            pipe.append(('scaler', StandardScaler()))
            pipe.append(('pca', PCA(n_components=self.n_components)))
        if self.mode == 'svm':
            pipe.append(('cls', SVC(C=self.C, kernel=self.kernel, probability=True)))
        elif self.mode == 'lda':
            pipe.append(('cls', LDA()))
        self.pipe = Pipeline(pipe)
        self.pipe.fit(X, y)

    def test(self, X, y):
        scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        preds = self.pipe.predict(X)
        # logits = self.pipe.predict_proba(X)
        scores['acc'] = accuracy_score(y, preds)
        scores['prec'] = precision_score(y, preds)
        scores['rec'] = recall_score(y, preds)
        scores['f1'] = f1_score(y, preds)
        return scores

    def cross_validate(self, X, y, cv=5, times=None):
        scores = {}
        times = 1 if times is None else times
        for _ in range(times):
            pid = np.random.permutation(len(y))
            X, y = X[pid], y[pid]
            cv = len(y) if cv == -1 else cv
            kf = KFold(n_splits=cv, shuffle=True)
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
        if mode == 'test':
            print(("{:<6}"*4).format('acc', 'prec', 'rec', 'f1'))
            table = []
            for _, val in scores.items():
                table.append("{:<6}".format("{:.2f}".format(val)))
            print(''.join(table))
        else:
            print(("{:<12}"*4).format('acc', 'prec', 'rec', 'f1'))
            table = []
            for _, val in scores.items():
                table.append("{:<12}".format("{:.2f}({:.2f})".format(val.mean(), val.std())))
            print(''.join(table))


if __name__ == "__main__":
    bertTrnDF = pd.read_csv('trn_bert_features.csv')
    bertTstDF = pd.read_csv('tst_bert_features.csv')
    bert_features, tst_bert_features = [], []
    for i in range(bertTrnDF.shape[0]):
        bert_features.append(bertTrnDF.iloc[i][-768:])
    bert_features = np.asarray(bert_features, dtype='float')
    for i in range(bertTstDF.shape[0]):
        tst_bert_features.append(bertTstDF.iloc[i][-768:])
    tst_bert_features = np.asarray(tst_bert_features, dtype='float')
    print('Shape: ', bert_features.shape, tst_bert_features.shape)
    print('BERT')
    svm = Classifier(mode='svm', pre='pca', C=1, kernel='linear')
    cv, times = 10, 10
    bert_scores, bert_tst_scores = svm.cv_test(
        bert_features, bertTrnDF.ad.values, tst_bert_features, bertTstDF.ad.values, 
        cv=cv, times=times, testtimes=times)
