import os
import torch
# import list and warnings
import warnings
from typing import List
import time
from unlearning_lib.metrics.auc_extended import auc_extended, balanced_auc_extended
from torch import nn, optim, default_generator, randperm
import math
import numpy as np

from torch.nn.utils import parameters_to_vector as p2v
import copy
from tqdm import tqdm
# from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from unlearning_lib.models.purchase_classifier import PurchaseClassifier
from unlearning_lib.methods.finetune import finetune
from unlearning_lib.methods.ascent import ascent
from unlearning_lib.methods.boundary_expanding import boundary_expanding
from unlearning_lib.methods.fisher_forgetting import fisher_forgetting
from unlearning_lib.methods.forsaken import forsaken
from unlearning_lib.methods.l_codec import L_CODEC
from unlearning_lib.methods.ssd import ssd

# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

def audit_score(model, unle_model, forget_set, retain_set, nonmem_set, shadow_set,
                model_numbs=20, shadow_path=None, DEVICE='cuda', PATH='LIRA_checkpoints/scores',
                SUFFIX='dataname_method_type_forgetsize', verbose=False, BALANCED=True, EXACT_FLAG=None, METRICS_FLAG=torch.ones(7), SEED=42):

    # create scores folder
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    dataname = SUFFIX.split('_')[0]
    if not os.path.exists(PATH+'/'+dataname):
        os.makedirs(PATH+'/'+dataname)

    if EXACT_FLAG is None:
        EXACT_FLAG = True if SUFFIX.split('_')[1] == 'retrain' else False

    Scores = {}
    print('**************************SEED**************************', SEED)
    audit_pred, audit_descriptions = audit_nonmem(
        model, unle_model, forget_set, nonmem_set, shadow_set, model_numbers=model_numbs, shadow_path=shadow_path, DEVICE=DEVICE, EXACT_FLAG=EXACT_FLAG, METRICS_FLAG=METRICS_FLAG)
    audit_pred_rt, _ = audit_nonmem(
        model, unle_model, retain_set, nonmem_set, shadow_set, model_numbers=model_numbs, shadow_path=shadow_path, DEVICE=DEVICE, EXACT_FLAG=EXACT_FLAG, METRICS_FLAG=METRICS_FLAG)
    # calculate the auc score for non-membership inference
    for key in audit_pred.keys():
        if verbose:
            print(audit_descriptions[key])
            print(f'{key:<25} non-membership inference:')

        metric_diff_all = np.concatenate(
            [audit_pred[key], audit_pred_rt[key]])
        labels = np.concatenate(
            [np.ones_like(audit_pred[key]), np.zeros_like(audit_pred_rt[key])])
        if BALANCED:
            auc_score, acc_score, desired_fpr, desired_tpr = balanced_auc_extended(
                labels, metric_diff_all, verbose=verbose)
        else:
            auc_score, acc_score, desired_fpr, desired_tpr = auc_extended(
                labels, metric_diff_all, verbose=verbose)
        Scores[key] = np.concatenate(
            [[auc_score], [acc_score], desired_tpr])

    np.save(PATH + f'/{dataname}/audit_pred_dict_' +
            str(SEED) + f'{SUFFIX}_{model_numbs}.npy', audit_pred)
    np.save(PATH + f'/{dataname}/audit_pred_dict_rt_' +
            str(SEED) + f'{SUFFIX}_{model_numbs}.npy', audit_pred_rt)

    return Scores, audit_descriptions


def audit_score_values(model, unle_model, forget_set, retain_set, nonmem_set, shadow_set,
                LOOP=5, model_numbs=20, shadow_path=None, DEVICE='cuda', PATH='LIRA_checkpoints/scores',
                SUFFIX='dataname_method_type_forgetsize', RNG=None, verbose=False, BALANCED=True, EXACT_FLAG=None, METRICS_FLAG=torch.ones(7)):

    # create scores folder
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    dataname = SUFFIX.split('_')[0]
    if not os.path.exists(PATH+'/'+dataname):
        os.makedirs(PATH+'/'+dataname)

    if EXACT_FLAG is None:
        EXACT_FLAG = True if SUFFIX.split('_')[1] == 'retrain' else False

    Scores = {}
    for loop in range(LOOP):
        Scores[loop] = {}
        print('**************************loop**************************', loop)
        audit_pred, audit_descriptions = audit_nonmem(
            model, unle_model, forget_set, nonmem_set, shadow_set, model_numbers=model_numbs, shadow_path=shadow_path, DEVICE=DEVICE, RNG=RNG, EXACT_FLAG=EXACT_FLAG, METRICS_FLAG=METRICS_FLAG)
        print('***********************50%')
        audit_pred_rt, _ = audit_nonmem(
            model, unle_model, retain_set, nonmem_set, shadow_set, model_numbers=model_numbs, shadow_path=shadow_path, DEVICE=DEVICE, RNG=RNG, EXACT_FLAG=EXACT_FLAG, METRICS_FLAG=METRICS_FLAG)
        print('***************************************************99%')
        # calculate the auc score for non-membership inference
        for key in audit_pred.keys():
            if verbose:
                print(audit_descriptions[key])
                print(f'{key:<25} non-membership inference:')

            metric_value_mean = audit_pred[key].mean()
            metric_value_std = audit_pred[key].std()
            metric_value_rt_mean = audit_pred_rt[key].mean()
            metric_value_rt_std = audit_pred_rt[key].std()

            Scores[loop][key] = np.concatenate(
                [[metric_value_mean], [metric_value_std], [metric_value_rt_mean], [metric_value_rt_std]])

    print_metric_value(Scores, audit_descriptions, LOOP=LOOP)


def update_audit_score(LOOP=5, model_numbs=20, shadow_path=None, DEVICE='cuda', PATH='LIRA_checkpoints/scores', SUFFIX='dataname_method_type_forgetsize', RNG=None, verbose=True):
    # create scores folder
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    dataname = SUFFIX.split('_')[0]
    if not os.path.exists(PATH+'/'+dataname):
        os.makedirs(PATH+'/'+dataname)
    Scores = {}
    for loop in range(LOOP):
        Scores[loop] = {}
        audit_pred_dict, audit_descriptions = np.load(PATH + f'/{dataname}/audit_pred_dict_' + str(
            loop) + f'{SUFFIX}_{LOOP}_{model_numbs}.npy', allow_pickle=True).item()
        audit_pred_dict_rt, _ = np.load(PATH + f'/{dataname}/audit_pred_dict_rt_' + str(
            loop) + f'{SUFFIX}_{LOOP}_{model_numbs}.npy', allow_pickle=True).item()
        # based on the dict values: audit_pred_dict = {'p_nm4n_m4o': p_nm4n_m4o, 'p_nm4n_m4o_lira': p_nm4n_m4o_lira, 'p_m4o_nm4n': p_m4o_nm4n,
        #    'p_m4o_nm4n_lira': p_m4o_nm4n_lira, 'p_nm4n': p_nm4n, 'p_nm4o': p_nm4o, 'p_update': p_update, 'p_after': p_after, 'p_before': p_before}
        # to get the variables
        for key in audit_pred_dict.keys():
            if verbose:
                print(audit_descriptions[key])
                print(f'{key:<25} non-membership inference:')
            metric_diff_all = np.concatenate(
                [audit_pred_dict[key], audit_pred_dict_rt[key]])
            labels = np.concatenate(
                [np.ones_like(audit_pred_dict[key]), np.zeros_like(audit_pred_dict_rt[key])])
            auc_score, acc_score, desired_fpr, desired_tpr = auc_extended(
                labels, metric_diff_all, verbose=verbose)
            Scores[loop][key] = np.concatenate(
                [[auc_score], [acc_score], desired_tpr])

    np.save(
        PATH + f'/{dataname}/scores_all_{SUFFIX}_{LOOP}_{model_numbs}.npy', Scores)
    print_auc(Scores, audit_descriptions, LOOP=LOOP)


def print_auc(Scores, audit_descriptions, LOOP=5, PRINT_DESCRIPTION=False):
    # calculate the mean and std of the scores
    scores_mean = {}
    scores_std = {}
    for key in Scores[0].keys():
        scores_mean[key] = [np.mean([Scores[loop][key][idx] for loop in range(
            LOOP)]) for idx in range(len(Scores[0][key]))]
        scores_std[key] = [np.std([Scores[loop][key][idx] for loop in range(
            LOOP)]) for idx in range(len(Scores[0][key]))]

    if PRINT_DESCRIPTION:
        for key in audit_descriptions.keys():
            print(f'{key:<25}: {audit_descriptions[key]}')

    # print the mean and std of the scores with keys
    print(f'{"Metric values mean":<30}', f'{"auc":<6}', f'{"acc":<6}',f'{"0.0":<6}', f'{".-5":<6}',
          f'{".-4":<6}', f'{".-3":<6}', f'{".-2":<6}', f'{".-1":<6}', f'{".-0":<6}')
    for key in scores_mean.keys():
        print(f'{key:<30}', " ".join(f"{x:.4f}" for x in scores_mean[key]))

    print('\n'+f'{"Metric values std":<30}', f'{"auc":<6}', f'{"acc":<6}', f'{"0.0":<6}', f'{".-5":<6}',
          f'{".-4":<6}', f'{".-3":<6}', f'{".-2":<6}', f'{".-1":<6}', f'{".-0":<6}')
    for key in scores_std.keys():
        print(f'{key:<30}', " ".join(f"{x:.4f}" for x in scores_std[key]))

    print('end_print\n\n')

def print_metric_value(Scores, audit_descriptions, LOOP=5, PRINT_DESCRIPTION=False):
    # calculate the mean and std of the scores
    scores_mean = {}
    scores_std = {}
    for key in Scores[0].keys():
        scores_mean[key] = [np.mean([Scores[loop][key][idx] for loop in range(
            LOOP)]) for idx in range(len(Scores[0][key]))]
        scores_std[key] = [np.std([Scores[loop][key][idx] for loop in range(
            LOOP)]) for idx in range(len(Scores[0][key]))]

    if PRINT_DESCRIPTION:
        for key in audit_descriptions.keys():
            print(f'{key:<25}: {audit_descriptions[key]}')

    # print the mean and std of the scores with keys
    print(f'{"Scores/tpr under fpr.xx":<30}', f'{"metric_value_mean":<6}', f'{"metric_value_std":<6}', f'{"metric_value_rt_mean":<6}',
          f'{"metric_value_rt_std":<6}')
    for key in scores_mean.keys():
        print(f'{key:<30}', " ".join(f"{x:.4f}" for x in scores_mean[key]))

    print('end_print\n\n')


def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(
                    f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        # type: ignore[arg-type]
        remainder = len(dataset) - sum(subset_lengths)
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")

    # type: ignore[call-overload]
    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def get_unlearn_model(model_ul, dataset_name, unlearn_method, retain_loader, forget_loader, val_loader, test_loader, class_num, DEVICE):
    if unlearn_method == 'finetune':
        model_ul = finetune(model_ul, retain_loader,
                            lr=1e-2, DEVICE=DEVICE)
    elif unlearn_method == 'ssd':
        if dataset_name in [ 'cifar10', 'cifar100']:
            alpha = 60
        else:
            alpha = 10
        model_ul = ssd(model_ul, retain_loader,
                       forget_loader, lr=1e-2, alpha=alpha, DEVICE=DEVICE)
    elif unlearn_method == 'l_codec':
        model_ul = L_CODEC(model_ul, forget_loader,
                           val_loader, dataset_name=dataset_name, lr=1e-1, DEVICE=DEVICE)
    elif unlearn_method == 'fisher':
        model_ul = fisher_forgetting(
            model_ul, retain_loader, DEVICE=DEVICE)
    elif unlearn_method == 'ascent':
        model_ul = ascent(model_ul, forget_loader, lr=1e-2, DEVICE=DEVICE)
    elif unlearn_method == 'boundary_expanding':
        model_ul = boundary_expanding(
            dataset_name, model_ul, forget_loader, class_num, DEVICE=DEVICE)
    elif unlearn_method == 'forsaken':
        model_ul = forsaken(model_ul, forget_loader,
                            retain_loader, test_loader, DEVICE=DEVICE)
    return model_ul