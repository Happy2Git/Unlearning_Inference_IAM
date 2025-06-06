import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .feature_builder import logit_scaling, model_confidence, predict_proba


def mia_unleak(old_model, new_model, target_loader, nonmem_loader, DEVICE=torch.device('cpu'), model_type='LR', model_numbers=20):
    old_model.eval()
    new_model.eval()
    # train attack models on nonmem set
    attack_model = train_attack_model(
        nonmem_loader, model_type=model_type, model_numbers=model_numbers)
    # obtain the posterior difference between the shadow model and the target model for target loader
    target_old_prob, target_new_prob = [], []
    for x, y in target_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        target_old_prob.append(predict_proba(old_model, x))
        target_new_prob.append(predict_proba(new_model, x))

    target_old_prob = torch.cat(target_old_prob)
    target_new_prob = torch.cat(target_new_prob)

    target_leak_feature = construct_leak_feature(
        target_old_prob, target_new_prob)
    pred = attack_model.predict_proba(target_leak_feature.cpu().numpy())

    return pred[:, 1]


def construct_leak_feature(target_old_prob, target_new_prob):
    # construct feature vector for target set ["direct_diff", "sorted_diff", 'direct_concat', 'sorted_concat', 'l2_distance', 'basic_mia']
    # unsorted feature
    direct_diff = target_old_prob - target_new_prob
    l2_distance = torch.norm(direct_diff, dim=1).unsqueeze(1)
    direct_concat = torch.cat([target_old_prob, target_new_prob], dim=1)
    basic_mia = target_old_prob
    # sorted feature
    target_old_sort, target_new_sort = [], []
    for index, pos in enumerate(target_old_prob):
        sort_indices = torch.argsort(pos, descending=False)
        target_old_sort.append(pos[sort_indices])
        target_new_sort.append(target_new_prob[index][sort_indices])

    target_old_sort = torch.stack(target_old_sort)
    target_new_sort = torch.stack(target_new_sort)
    sorted_diff = target_old_sort - target_new_sort
    sorted_concat = torch.cat([target_old_sort, target_new_sort], dim=1)

    leak_feature = torch.cat(
        [direct_diff, sorted_diff, direct_concat, sorted_concat, l2_distance, basic_mia], dim=1)

    return leak_feature


def train_attack_model(nonmem_loader, model_type='LR', model_numbers=20):
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

    # attack model
    attack_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000))
    X_shadows, y_shadows = [], []
    for i in range(model_numbers):
        # shadow models: mimic the origin model and unlearned model
        origin_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000))
        unlearn_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000))
        pos_indices = np.random.choice(
            X_all.shape[0], int(X_all.shape[0]*0.8), replace=False)
        neg_indices = np.setdiff1d(np.arange(X_all.shape[0]), pos_indices)
        # generate target indices from pos_indices
        target_indices = np.random.choice(pos_indices, int(
            pos_indices.shape[0]*0.5), replace=False)
        non_target_indices = np.setdiff1d(pos_indices, target_indices)

        pos_features = X_all[pos_indices]
        pos_labels = y_all[pos_indices]
        neg_features = X_all[neg_indices]
        neg_labels = y_all[neg_indices]
        target_features = X_all[target_indices]
        target_labels = y_all[target_indices]
        non_target_features = X_all[non_target_indices]
        non_target_labels = y_all[non_target_indices]
        origin_model.fit(pos_features, pos_labels)
        unlearn_model.fit(non_target_features, non_target_labels)
        # collect training data for attack model
        target_old_prob = origin_model.predict_proba(target_features)
        target_new_prob = unlearn_model.predict_proba(target_features)
        target_pos_feature = construct_leak_feature(torch.from_numpy(
            target_old_prob), torch.from_numpy(target_new_prob))
        neg_old_prob = origin_model.predict_proba(neg_features)
        neg_new_prob = unlearn_model.predict_proba(neg_features)
        other_neg_feature = construct_leak_feature(
            torch.from_numpy(neg_old_prob), torch.from_numpy(neg_new_prob))

        target_pos_labels = np.ones((target_pos_feature.shape[0]))
        other_neg_labels = np.zeros((other_neg_feature.shape[0]))

        X_shadow = np.concatenate([target_pos_feature, other_neg_feature])
        y_shadow = np.concatenate([target_pos_labels, other_neg_labels])

        X_shadows.append(X_shadow)
        y_shadows.append(y_shadow)

    X_shadows = np.concatenate(X_shadows, axis=0)
    y_shadows = np.concatenate(y_shadows, axis=0)
    attack_model.fit(X_shadows, y_shadows)
    return attack_model
