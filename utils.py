import os
from glob import glob
import torch
import regex as re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

data_path = '../ADReSS-IS2020-data'

##################################################
# Extract Information and text in *.cha
##################################################


def extract_pid(file_name):
    par = {}
    par['id'] = file_name.split('/')[-1].split('.cha')[0]
    f = iter(open(file_name, encoding='utf-8'))
    l = next(f)
    while (True):
        if l.startswith('@PID:'):
            par['pid'] = l.replace('@PID:\t', '').replace('\n', '')
            break
        l = next(f)
    return par


def extract_cha(file_name):
    par = {}
    par['id'] = file_name.split('/')[-1].split('.cha')[0]
    f = iter(open(file_name))
    l = next(f)
    trans = []
    try:
        curr_trans = ''
        while (True):
            if l.startswith('@ID'):
                participant = [i.strip() for i in l.split('|')]
                if participant[2] == 'PAR':
                    try:
                        par['mmse'] = '' if len(
                            participant[8]) == 0 else float(participant[8])
                    except:
                        par['mmse'] = ''
                    par['sex'] = participant[4][0]
                    par['age'] = int(participant[3].replace(';', ''))
            if l.startswith('*PAR:') or l.startswith('*INV'):
                curr_trans = l
            elif len(curr_trans) != 0 and not(l.startswith('%') or l.startswith('*')):
                curr_trans += l
            elif len(curr_trans) > 0:
                trans.append(curr_trans)
                curr_trans = ''
            l = next(f)
    except StopIteration:
        pass

    clean_par_trans = []
    clean_all_trans = []
    par_trans_time_segments = []
    all_trans_time_segments = []
    is_par = False
    for s in trans:
        def _parse_time(s):
            return [*map(int, re.search(r'\x15(\d*_\d*)\x15', s).groups()[0].split('_'))]

        def _clean(s):
            s = re.sub(r'\x15\d*_\d*\x15', '', s)  # remove time block
            s = re.sub(r'\[.*\]', '', s)  # remove other trans artifacts [.*]
            s = s.strip()
            # remove tab, new lines, inferred trans??, ampersand, &
            s = re.sub(r'\t|\n|<|>', '', s)
            return s

        if s.startswith('*PAR:'):
            is_par = True
        elif s.startswith('*INV:'):
            is_par = False
            s = re.sub(r'\*INV:\t', '', s)  # remove prefix

        if is_par:
            s = re.sub(r'\*PAR:\t', '', s)  # remove prefix
            par_trans_time_segments.append(_parse_time(s))
            clean_par_trans.append(_clean(s))
        all_trans_time_segments.append(_parse_time(s))
        clean_all_trans.append(_clean(s))

    par['trans'] = trans
    par['clean_trans'] = clean_all_trans
    par['clean_par_trans'] = clean_par_trans
    par['joined_all_trans'] = ' '.join(clean_all_trans)
    par['joined_all_par_trans'] = ' '.join(clean_par_trans)

    # sentence times
    par['par_trans_time_segments'] = par_trans_time_segments
    # par['per_sent_times'] = [par_trans_time_segments[i][1] - par_trans_time_segments[i][0]
    #                          for i in range(len(par_trans_time_segments))]
    # par['total_time'] = par_trans_time_segments[-1][1] - \
    #     par_trans_time_segments[0][0]
    # par['time_before_par_trans'] = par_trans_time_segments[0][0]
    # par['time_between_sents'] = [0 if i == 0 else max(0, par_trans_time_segments[i][0] - par_trans_time_segments[i-1][1])
    #                              for i in range(len(par_trans_time_segments))]
    return par


def _parse_data(data_dir):
    prob_ad_dir = f'{data_dir}/transcription/cd/*.cha'
    controls_dir = f'{data_dir}/transcription/cc/*.cha'

    prob_ad = [extract_cha(fn) for fn in glob(prob_ad_dir)]
    controls = [extract_cha(fn) for fn in glob(controls_dir)]
    controls_df = pd.DataFrame(controls)
    prob_ad_df = pd.DataFrame(prob_ad)
    controls_df['ad'] = 0
    prob_ad_df['ad'] = 1
    df = pd.concat([controls_df, prob_ad_df]).sample(
        frac=1).reset_index(drop=True)
    return df


def parse_train_data(data_path):
    return _parse_data(data_path+'/train')


def parse_test_data(data_path):
    return pd.DataFrame([extract_cha(fn) for fn in glob(data_path+'/test/transcription/*.cha')])


def extract_utts(clean_par_trans, ad):
    utts = []
    ads = []
    for i in range(len(ad)):
        tmp = re.findall(r'\"(.+?)\"', clean_par_trans[i])
        utts.extend(tmp)
        ads.extend([ad[i]] * len(tmp))
    return utts, ads


##################################################
# Plot
##################################################


def plot_roc_std(scores, ax, style, title, fill=False):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(scores['fprs'].shape[0]):
        interp_tpr = np.interp(mean_fpr, scores['fprs'][i], scores['tprs'][i])
        # interp_tpr = np.interp(mean_fpr, scores['fprs'][i][0], scores['tprs'][i][0])
        # interp_tpr = np.interp(mean_fpr, scores['fprs'][i][0][0], scores['tprs'][i][0][0])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(scores['auc'])
    std_auc = np.std(scores['auc'])
    ax.plot(mean_fpr, mean_tpr, dashes=style,
            label=title+r' $({:.2f}\pm{:.2f})$'.format(mean_auc, std_auc),
            lw=2, alpha=.8)
    if fill:
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                        alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel='Specificity (FPR)', ylabel='Sensitivity (TPR)')
    ax.legend(handlelength=1.8, loc="lower right")
    return ax
