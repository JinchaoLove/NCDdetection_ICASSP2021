import os
import sys
import importlib
import librosa
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import transformers as ppb

from models import *
from modules import *
# 'ad' : control - 0, ad - 1
# 'sex': male - 0, female - 1
data_path = '../ADReSS-IS2020-data'
trans, wavs, pittpar = '/transcription/', '/Full_wave_enhanced_audio/', '/pittpar/'
# %%
# ComParE
train_wavs = pd.read_csv('feats/train_compare_pittpar.csv')  # [5:] # pittpar
test_wavs = pd.read_csv('feats/test_compare_pittpar.csv')  # [3:] # full_wavs

train_compares = []
for i in range(train_wavs.shape[0]):
    train_compares.append(train_wavs.iloc[i][-6373:])
train_compares = np.asarray(train_compares)

test_compares = []
for i in range(test_wavs.shape[0]):
    test_compares.append(test_wavs.iloc[i][-6373:])
test_compares = np.asarray(test_compares)

# Linguistic
train_cc_eval = pd.read_csv(data_path+'/train/transcription/cc.eval.csv')
train_cd_eval = pd.read_csv(data_path+'/train/transcription/cd.eval.csv')
test_eval = pd.read_csv(data_path+'/test/test.eval.csv')
train_ling, ling_ad = [], []
for i in range(train_cc_eval.shape[0]):
    train_ling.append(train_cc_eval.iloc[i][-34:])
    ling_ad.append(0)
for i in range(train_cd_eval.shape[0]):
    train_ling.append(train_cd_eval.iloc[i][-34:])
    ling_ad.append(1)
train_ling = np.asarray(train_ling, dtype=np.float)
ling_ad = np.asarray(ling_ad)
test_ling = []
for i in range(test_eval.shape[0]):
    test_ling.append(test_eval.iloc[i][-34:])
test_ling = np.asarray(test_ling, dtype=np.float)

# test_labels
test_res = pd.read_csv(data_path+'/test/test_meta_data.txt', delimiter=';')
test_res['ID   '] = test_res['ID   '].apply(lambda x: x.strip(' '))
test_compare_ad = np.zeros(len(test_wavs))
for i in range(len(test_wavs)):
    idx = test_res['ID   '] == test_wavs.loc[i, 'id'][:4]
    test_compare_ad[i] = test_res.loc[idx, 'Label ']
test_ling_ad = np.zeros(len(test_eval))
for i in range(len(test_eval)):
    idx = test_res['ID   '] == test_eval.loc[i, 'File'][:-4]
    test_ling_ad[i] = test_res.loc[idx, 'Label ']

# Pearson's correlation test:
# remove |R| > 0.2 correlated with duration


def corr_select(feats, ref, thred=0.2):
    selects = []
    ps = []
    for i in range(feats.shape[1]):
        corr, _ = pearsonr(feats[:, i], ref)
        # if np.abs(corr) < thred:
        if np.abs(corr) > thred:
            selects.append(i)
            ps.append(np.abs(corr))
    return selects, ps


com_idx, com_ps = corr_select(train_compares, train_wavs.ad.values, thred=0.2)
p_train_compares = train_compares[:, com_idx]
p_test_compares = test_compares[:, com_idx]
ling_idx, ling_ps = corr_select(train_ling, ling_ad, thred=0.2)
p_train_ling = train_ling[:, ling_idx]
p_test_ling = test_ling[:, ling_idx]
print('After selection:\n  ComParE: train {}, test {}\n  Linguistic: train {}, test {}'.format(
    p_train_compares.shape, p_test_compares.shape, p_train_ling.shape, p_test_ling.shape))

feats_name_df = pd.read_csv('test_pit_comps.csv', header=None, skiprows=3, usecols=[0])
feats_name = feats_name_df[0].apply(lambda x: x.replace(
    '@attribute ', '').replace(' numeric', '')).values[:6373]
com_ps = np.array(com_ps)
k = 10
print('p: ', com_ps[com_ps.argsort()[-k:][::-1]])
print('idx: ', com_ps.argsort()[-k:][::-1])
print('top 10 feats: ', feats_name[com_ps.argsort()[-k:][::-1]])

# %%
# Xvec
train_cc_xvecs = np.load('feats/xvec_pittpar_cc.npz', allow_pickle=True)
train_cd_xvecs = np.load('feats/xvec_pittpar_cd.npz', allow_pickle=True)
test_xvecs0 = np.load('feats/xvec_pittpar_test.npz', allow_pickle=True)  # ['data_path', 'features']
train_xvecs = np.concatenate([train_cc_xvecs['features'], train_cd_xvecs['features']], axis=0)
train_xvec_labels = np.concatenate(
    [np.zeros(train_cc_xvecs['features'].shape[0]), np.ones(train_cd_xvecs['features'].shape[0])], axis=0)
test_xvecs = test_xvecs0['features']
test_xvec_labels = test_res['Label '].values

# %%
# BERT
train_df = pd.read_csv('feats/train_chas.csv')
test_df = pd.read_csv('feats/test_chas.csv')
test_df = test_df.sort_values(by=['id']).reset_index(drop=True)
test_ad = np.zeros(len(test_df))
for i in range(len(test_df)):
    idx = test_res['ID   '] == test_df.loc[i, 'id']
    test_ad[i] = test_res.loc[idx, 'Label ']
model_class, tokenizer_class, pretrained_weights = (
    ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer, model = load_transformer_model_tokenizer(
    model_class, tokenizer_class, pretrained_weights)
bert_features, _ = bert_embed1d(train_df.joined_all_par_trans.values, tokenizer, model)
tst_bert_features, _ = bert_embed1d(test_df.joined_all_par_trans.values, tokenizer, model)

model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer, model = load_transformer_model_tokenizer(
    model_class, tokenizer_class, pretrained_weights)
disbert_features, _ = bert_embed1d(train_df.joined_all_par_trans.values, tokenizer, model)
tst_disbert_features, _ = bert_embed1d(test_df.joined_all_par_trans.values, tokenizer, model)
bert_labels = train_df.ad.values
tst_bert_labels = test_ad

# %%
cv, times = 10, 10
print('# CRNN')
print('pComParE')
clf = Classifier(
    module=CRNN1D,
    module_args={'input_dim': p_train_compares.shape[-1], 'hidden_size': 128, 'dropout': 0.2, },
    # optimizer=ppb.Adafactor,
    optimizer=torch.optim.AdamW,
    opt_args={'lr': 5e-4, 'weight_decay': 1e-2},
    train_args={'batch_size': 32, 'max_epochs': 200, 'earlystop': 10, 'verbose': 1,
                # 'normalize':False, # False for TfidfVectorizer
                # 'pca': True, # True for PCA
                'expand_dim': True,
                },
)
pcompare_scores, pcompare_tst_scores = clf.cv_test(
    p_train_compares, train_wavs.ad.values, p_test_compares, test_compare_ad, cv=cv, times=times, testtimes=times)

print('ling')
clf = Classifier(
    module=CRNN1D,
    module_args={'input_dim': train_ling.shape[-1], 'hidden_size': 128, 'dropout': 0.2, },
    # optimizer=ppb.Adafactor,
    optimizer=torch.optim.AdamW,
    opt_args={'lr': 5e-4, 'weight_decay': 1e-2},
    train_args={'batch_size': 32, 'max_epochs': 200, 'earlystop': 10, 'verbose': 1,
                # 'normalize':False, # False for TfidfVectorizer
                # 'pca': True, # True for PCA
                'expand_dim': True,
                },
)
ling_scores, ling_tst_scores = clf.cv_test(
    train_ling, ling_ad, test_ling, test_ling_ad, cv=cv, times=times, testtimes=times)

print('xvec')
clf = Classifier(
    module=CRNN1D,
    module_args={'input_dim': train_xvecs.shape[-1], 'hidden_size': 128, 'dropout': 0.2, },
    # optimizer=ppb.Adafactor,
    optimizer=torch.optim.AdamW,
    opt_args={'lr': 5e-4, 'weight_decay': 1e-2},
    train_args={'batch_size': 32, 'max_epochs': 200, 'earlystop': 10, 'verbose': 1,
                # 'normalize':False, # False for TfidfVectorizer
                # 'pca': True, # True for PCA
                'expand_dim': True,
                },
)
xvec_scores, xvec_tst_scores = clf.cv_test(
    train_xvecs, train_xvec_labels, test_xvecs, test_xvec_labels, cv=cv, times=times, testtimes=times)

print('TFIDF')
vect = TfidfVectorizer() # use_idf=True, smooth_idf=True, sublinear_tf=False
train_tfs, train_tf_labels = vect.fit_transform(train_df.joined_all_par_trans.values).toarray(), train_df.ad.values
tst_tfs = vect.transform(test_df.joined_all_par_trans.values).toarray()
tst_tf_labels = test_ad
clf = Classifier(
    module=CRNN1D,
    module_args={'input_dim': train_tfs.shape[-1], 'hidden_size': 128, 'dropout': 0.2, },
    # optimizer=ppb.Adafactor,
    optimizer=torch.optim.AdamW,
    opt_args={'lr': 5e-4, 'weight_decay': 1e-2},
    train_args={'batch_size': 32, 'max_epochs': 200, 'earlystop': 10, 'verbose': 1,
                'normalize':False, # False for TfidfVectorizer
                # 'pca': True, # True for PCA
                'expand_dim': True,
                },
)
tfidf_scores, tfidf_tst_scores = clf.cv_test(
    train_tfs, train_tf_labels, tst_tfs, tst_tf_labels, cv=cv, times=times, testtimes=times)

print('BERT')
clf = Classifier(
    module=CRNN1D,
    module_args={'input_dim': bert_features.shape[-1], 'hidden_size': 128, 'dropout': 0.2, },
    # optimizer=ppb.Adafactor,
    optimizer=torch.optim.AdamW,
    opt_args={'lr': 5e-4, 'weight_decay': 1e-2},
    train_args={'batch_size': 32, 'max_epochs': 200, 'earlystop': 10, 'verbose': 1,
                # 'normalize':False, # False for TfidfVectorizer
                # 'pca': True, # True for PCA
                'expand_dim': True,
                },
)
bert_scores, bert_tst_scores = clf.cv_test(
    bert_features, bert_labels, tst_bert_features, tst_bert_labels, cv=cv, times=times, testtimes=times)

print('DistilBERT')
clf = Classifier(
    module=CRNN1D,
    module_args={'input_dim': disbert_features.shape[-1], 'hidden_size': 128, 'dropout': 0.2, },
    # optimizer=ppb.Adafactor,
    optimizer=torch.optim.AdamW,
    opt_args={'lr': 5e-4, 'weight_decay': 1e-2},
    train_args={'batch_size': 32, 'max_epochs': 200, 'earlystop': 10, 'verbose': 1,
                # 'normalize':False, # False for TfidfVectorizer
                # 'pca': True, # True for PCA
                'expand_dim': True,
                },
)
distilbert_scores, distilbert_tst_scores = clf.cv_test(
    disbert_features, bert_labels, tst_disbert_features, tst_bert_labels, cv=cv, times=times, testtimes=times)

# %%
scores = [pcompare_scores, ling_scores, xvec_scores, tfidf_scores, bert_scores, distilbert_scores]
tst_scores = [pcompare_tst_scores, ling_tst_scores, xvec_tst_scores,
              tfidf_tst_scores, bert_tst_scores, distilbert_tst_scores]
np.savez('crnn_scores.npz', scores)
np.savez('crnn_tst_scores.npz', tst_scores)
