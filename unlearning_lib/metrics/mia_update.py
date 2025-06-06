import torch
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from .feature_builder import logit_scaling, model_confidence


def mia_update(old_model, new_model, target_loader, nonmem_loader, DEVICE=torch.device('cpu'), model_type='LR', model_numbers=20):
    # estimate the mean and variance of the nonmem set for old model
    old_model.eval()
    new_model.eval()
    # calculate the scaled logit of the model confidence for each example in the non-membership set
    mean_shadows, var_shadows = micmic_means_vars(
        nonmem_loader, model_type=model_type, model_numbers=model_numbers)

    # calculate the scaled logit of the model confidence for each example in the target set
    target_old_conf, target_new_conf = [], []
    for x, y in target_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        target_old_conf.append(logit_scaling(
            model_confidence(old_model, x, y)))
        target_new_conf.append(logit_scaling(
            model_confidence(new_model, x, y)))

    metric_update = []
    for index in range(len(target_old_conf)):
        cdf_scores = []
        for mean_shadow, var_shadow in zip(mean_shadows, var_shadows):
            # norm.cdf(target, loc=mean, scale=var+1e-30): probability of the target located at the left of mean
            cdf_score_before = 1 - \
                norm.cdf(target_old_conf[index],
                         loc=mean_shadow, scale=var_shadow+1e-30)
            cdf_score_after = 1 - \
                norm.cdf(target_new_conf[index],
                         loc=mean_shadow, scale=var_shadow+1e-30)
            cdf_scores.append(cdf_score_after-cdf_score_before)
        metric_update.append(np.mean(cdf_scores, axis=0))

    metric_update = np.concatenate(metric_update)

    return metric_update


def micmic_means_vars(nonmem_loader, model_type='LR', model_numbers=20):
    '''
    train attack model on nonmem set
    '''
    features_list = []
    labels_list = []
    for batch in nonmem_loader:
        features, labels = batch
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        if features_np.ndim > 2:
            features_np = features_np.reshape(features_np.shape[0], -1)

        features_list.append(features_np)
        labels_list.append(labels_np)
    X_all = np.concatenate(features_list, axis=0)
    y_all = np.concatenate(labels_list, axis=0)
    train_indices = np.random.choice(
        X_all.shape[0], int(0.6*X_all.shape[0]), replace=False)
    test_indices = np.setdiff1d(np.arange(X_all.shape[0]), train_indices)
    test_features = X_all[test_indices]
    test_labels = y_all[test_indices]

    mean_shadows, var_shadows = [], []
    for i in range(model_numbers):
        # shadow models: mimic the origin model and unlearned model
        shadow_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000))
        pos_indices = np.random.choice(train_indices, int(
            len(train_indices)*0.6), replace=False)
        pos_features = X_all[pos_indices]
        pos_labels = y_all[pos_indices]
        shadow_model.fit(X_all, y_all)
        # collect training data for attack model
        pos_prob = shadow_model.predict_proba(test_features)
        # get the corresponding label confidence for test_features, with test_labels as the label
        pos_conf = pos_prob[np.arange(test_features.shape[0]), test_labels]
        pos_conf = logit_scaling(torch.from_numpy(pos_conf))
        mean_shadow = torch.mean(pos_conf)
        var_shadow = torch.var(pos_conf)
        mean_shadows.append(mean_shadow)
        var_shadows.append(var_shadow)

    return mean_shadows, var_shadows
