import numpy as np
from sklearn.metrics import roc_curve, auc


def balanced_auc_extended(labels, metric_diff_all, iterations=5000, verbose=True):
    """
    Calculate the AUC score and true positive rate at a low false positive rate for the non-membership inference attack.
    """
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    sample_size = int(min(len(positive_indices), len(negative_indices)) * 0.8)

    interp_tpr_list, interp_tnr_list = [], []
    auc_list, acc_list = [], []

    for _ in range(iterations):
        # Randomly sample indices from positive and negative classes to create a balanced dataset
        sampled_positive_indices = np.random.choice(positive_indices, size=sample_size, replace=True)
        sampled_negative_indices = np.random.choice(negative_indices, size=sample_size, replace=True)
        
        # Combine the resampled indices
        resampled_indices = np.concatenate((sampled_positive_indices, sampled_negative_indices))
        # Extract the corresponding labels and predictions
        resampled_labels = labels[resampled_indices]
        resampled_metric_diff_all = metric_diff_all[resampled_indices]    
        # Calculate the ROC curve for the balanced dataset
        fpr, tpr, _ = roc_curve(resampled_labels, resampled_metric_diff_all)
        auc_score = auc(fpr, tpr)
        acc_score = np.max(1-(fpr+(1-tpr))/2)
    
        # Define the specific FPR and FNR thresholds
        low_fpr_thresholds = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        low_fnr_thresholds = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

        # Calculate FNR and TNR
        fnr = 1 - tpr
        tnr = 1 - fpr

        # Interpolate TPR at specific FPR thresholds
        tpr_at_low_fpr = np.interp(low_fpr_thresholds, fpr, tpr)

        # Interpolate TNR at specific FNR thresholds
        # Since FNR decreases as TPR increases, we need to reverse the arrays for proper interpolation
        sorted_fnr_indices = np.argsort(fnr)
        fnr_sorted = fnr[sorted_fnr_indices]
        tnr_sorted = tnr[sorted_fnr_indices]

        tnr_at_low_fnr = np.interp(low_fnr_thresholds, fnr_sorted, tnr_sorted)

        interp_tpr_list.append(tpr_at_low_fpr)
        interp_tnr_list.append(tnr_at_low_fnr)

        auc_list.append(auc_score)
        acc_list.append(acc_score)

    interp_tpr_list = np.stack(interp_tpr_list)
    interp_tnr_list = np.stack(interp_tnr_list)

    # Calculate the average FPR and TPR
    acc_score = np.mean(acc_list)
    auc_score = np.mean(auc_list)

    interp_tpr_list = np.mean(interp_tpr_list, axis=0)
    interp_tnr_list = np.mean(interp_tnr_list, axis=0)

    return auc_score, acc_score, low_fpr_thresholds, interp_tpr_list, interp_tnr_list

def auc_extended(labels, metric_diff_all, verbose=True):
    """
    Calculate the AUC score and true positive rate at a low false positive rate for the non-membership inference attack.
    """
    fpr, tpr, _ = roc_curve(labels, metric_diff_all)
    auc_score = auc(fpr, tpr)
    acc_score = np.max(1-(fpr+(1-tpr))/2)
    # Define the specific FPR and FNR thresholds
    low_fpr_thresholds = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    low_fnr_thresholds = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    # Calculate FNR and TNR
    fnr = 1 - tpr
    tnr = 1 - fpr

    # Interpolate TPR at specific FPR thresholds
    tpr_at_low_fpr = np.interp(low_fpr_thresholds, fpr, tpr)

    # Interpolate TNR at specific FNR thresholds
    # Since FNR decreases as TPR increases, we need to reverse the arrays for proper interpolation
    sorted_fnr_indices = np.argsort(fnr)
    fnr_sorted = fnr[sorted_fnr_indices]
    tnr_sorted = tnr[sorted_fnr_indices]

    tnr_at_low_fnr = np.interp(low_fnr_thresholds, fnr_sorted, tnr_sorted)
 
    return auc_score, acc_score, low_fpr_thresholds, tpr_at_low_fpr, tnr_at_low_fnr

def balanced_auc_extended(labels, metric_diff_all, iterations=5000, verbose=True):
    """
    Calculate the AUC score and true positive rate at a low false positive rate for the non-membership inference attack.
    """
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    sample_size = int(min(len(positive_indices), len(negative_indices)) * 0.8)

    interp_tpr_list, interp_tnr_list = [], []
    auc_list, acc_list = [], []

    for _ in range(iterations):
        # Randomly sample indices from positive and negative classes to create a balanced dataset
        sampled_positive_indices = np.random.choice(positive_indices, size=sample_size, replace=True)
        sampled_negative_indices = np.random.choice(negative_indices, size=sample_size, replace=True)
        
        # Combine the resampled indices
        resampled_indices = np.concatenate((sampled_positive_indices, sampled_negative_indices))
        # Extract the corresponding labels and predictions
        resampled_labels = labels[resampled_indices]
        resampled_metric_diff_all = metric_diff_all[resampled_indices]    
        # Calculate the ROC curve for the balanced dataset
        fpr, tpr, _ = roc_curve(resampled_labels, resampled_metric_diff_all)
        auc_score = auc(fpr, tpr)
        acc_score = np.max(1-(fpr+(1-tpr))/2)
    
        # Define the specific FPR and FNR thresholds
        low_fpr_thresholds = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        low_fnr_thresholds = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

        # Calculate FNR and TNR
        fnr = 1 - tpr
        tnr = 1 - fpr

        # Interpolate TPR at specific FPR thresholds
        tpr_at_low_fpr = np.interp(low_fpr_thresholds, fpr, tpr)

        # Interpolate TNR at specific FNR thresholds
        # Since FNR decreases as TPR increases, we need to reverse the arrays for proper interpolation
        sorted_fnr_indices = np.argsort(fnr)
        fnr_sorted = fnr[sorted_fnr_indices]
        tnr_sorted = tnr[sorted_fnr_indices]

        tnr_at_low_fnr = np.interp(low_fnr_thresholds, fnr_sorted, tnr_sorted)

        interp_tpr_list.append(tpr_at_low_fpr)
        interp_tnr_list.append(tnr_at_low_fnr)

        auc_list.append(auc_score)
        acc_list.append(acc_score)

    interp_tpr_list = np.stack(interp_tpr_list)
    interp_tnr_list = np.stack(interp_tnr_list)

    # Calculate the average FPR and TPR
    acc_score = np.mean(acc_list)
    auc_score = np.mean(auc_list)

    interp_tpr_list = np.mean(interp_tpr_list, axis=0)
    interp_tnr_list = np.mean(interp_tnr_list, axis=0)

    return auc_score, acc_score, low_fpr_thresholds, interp_tpr_list, interp_tnr_list