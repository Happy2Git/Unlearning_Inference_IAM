import torch
import numpy as np
from scipy.stats import gumbel_r, norm # Replaced invgauss with norm (normal distribution)
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
import numpy as np
from scipy.stats import norm, beta
from tqdm import tqdm
from scipy import stats
from scipy.interpolate import interp1d

def kde_scores_all(interpolated_logloss, target_vals, bandwidth=0.1):

    n_inter, n_samples = interpolated_logloss.shape[1], interpolated_logloss.shape[2]
    scores = np.zeros((n_inter, n_samples))

    for j in tqdm(range(n_samples), desc="sample id"):
        for i in range(n_inter):
            obs_vals = interpolated_logloss[:, i, j].reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(obs_vals)

            def pdf_func(x):
                return np.exp(kde.score_samples([[x]]))[0]

            val = target_vals[i, j]
            try:
                scores[i, j] = quad(pdf_func, -np.inf, val, limit=100)[0]
            except:
                scores[i, j] = 0.0  # fallback for numerical failure

    return scores

def interpolate_appro(target_ori_signal, target_unl_signal, shadow_target_signal, INTER_APPRO=100, SHADOW_SIMU=128, type = 'gumbel', fix_variance=True, interpolate_type='linear'):
    INTER_APPRO = max(2, INTER_APPRO) # when INTER_APPRO is 2, it means no interpolation
    if target_ori_signal is None:
        eps1 = 1e-2
        eps2 = 1e-5
        target_ori_signal = (-np.log(eps1-np.log(eps2+1)))*torch.ones_like(target_unl_signal)

    if shadow_target_signal.shape[0]==1:
        fix_variance = True

    alphas = np.linspace(0, 1, INTER_APPRO)

    if interpolate_type == 'linear':
        interpolated_logloss = torch.stack([
            shadow_target_signal[:SHADOW_SIMU,:] + (target_ori_signal - shadow_target_signal[:SHADOW_SIMU,:]) * (alpha)
            for alpha in alphas
        ], dim=1)

    print('interpolated_logloss shape:', interpolated_logloss.shape)

    expand_target_unl = target_unl_signal.expand(INTER_APPRO-1, len(target_unl_signal)).flatten()
    if type == 'gumbel':
        mean_data = ((interpolated_logloss.numpy()).mean(axis=0)[:-1]).reshape(-1)
        if fix_variance:
            std_ = np.std(interpolated_logloss.numpy(), axis=(0, 2))[:-1]
            std_data = np.tile(np.expand_dims(std_, axis=1), (1,interpolated_logloss.shape[2])).reshape(-1)
        else:
            std_data =((interpolated_logloss.numpy()).std(axis=0)[:-1]).reshape(-1)
            std_data = np.where(std_data==0, np.full_like(std_data, 1e-10), std_data)
        # Estimate parameters using method of moments
        gamma = 0.5772  # Euler-Mascheroni constant
        beta_mom = np.sqrt(6) * std_data / np.pi
        mu_mom = mean_data - beta_mom * gamma
        score_unl_expa = gumbel_r.cdf(expand_target_unl, mu_mom, beta_mom)
    elif type == 'norm':
        mean_data = ((interpolated_logloss.numpy()).mean(axis=0)[:-1]).reshape(-1)
        if fix_variance:
            std_ = np.std(interpolated_logloss.numpy(), axis=(0, 2))[:-1]
            std_data = np.tile(np.expand_dims(std_, axis=1), (1,interpolated_logloss.shape[2])).reshape(-1)
        else:
            std_data =((interpolated_logloss.numpy()).std(axis=0)[:-1]).reshape(-1)

        score_unl_expa = norm.cdf(expand_target_unl, loc=mean_data, scale=std_data+1e-30)
    elif type == 'ecdf':
        interpolated_subset = interpolated_logloss[:, :-1, :].numpy()  # Shape: [SHADOW_SIMU, INTER_APPRO-1, data_points]
        target_unl_reshaped = expand_target_unl.reshape(INTER_APPRO-1, -1)
        score_unl_expa = (interpolated_subset <= np.expand_dims(target_unl_reshaped, 0)).mean(axis=0).flatten()
    elif type == 'beta':
        mean_beta = ((interpolated_logloss.numpy()).mean(axis=0)[:-1]).reshape(-1)
        if fix_variance:
            std_ = np.std(interpolated_logloss.numpy(), axis=(0, 2))[:-1]
            std_data = np.tile(np.expand_dims(std_, axis=1), (1,interpolated_logloss.shape[2])).reshape(-1)
        else:
            std_data =((interpolated_logloss.numpy()).std(axis=0)[:-1]).reshape(-1)
        var_beta = std_data ** 2

        var_beta = np.where(var_beta == 0, 1e-10, var_beta)
        mean_clipped = np.clip(mean_beta, 1e-10, 1 - 1e-10)
        
        temp = (mean_clipped * (1 - mean_clipped)) / var_beta - 1
        temp = np.maximum(temp, 1e-10)  # Ensure positivity
        
        alpha_param = mean_clipped * temp
        beta_param = (1 - mean_clipped) * temp
        
        alpha_param = np.maximum(alpha_param, 1e-10)
        beta_param = np.maximum(beta_param, 1e-10)
        
        expand_target_unl_np = expand_target_unl.numpy().flatten()
        score_unl_expa = stats.beta.cdf(expand_target_unl_np, alpha_param, beta_param)
    elif type == 'kde':
        # Parameters
        n_samples = interpolated_logloss.shape[0]
        if fix_variance:
            std_ = np.std(interpolated_logloss.numpy(), axis=(0, 2))[:-1]
            std_data = np.tile(np.expand_dims(std_, axis=1), (1,interpolated_logloss.shape[2])).reshape(-1)
        else:
            std_data =((interpolated_logloss.numpy()).std(axis=0)[:-1]).reshape(-1)

        interpolated_logloss_kde = interpolated_logloss[:, :-1, :].reshape(n_samples, -1).numpy() 
        interpolated_logloss_kde = interpolated_logloss_kde.T  # shape: (10000, 20)
        n_datasets = interpolated_logloss_kde.shape[0]
        
        h = std_data * (1 ** (-1/5))  # Bandwidth for each row
        expand_target_unl = expand_target_unl.numpy().reshape(-1,1)
        h = h.reshape(-1,1)
        score_unl_expa = norm.cdf((expand_target_unl - interpolated_logloss_kde) / h).mean(axis=1)

    score_unl_expa = score_unl_expa.reshape(INTER_APPRO-1, -1)
    appros = alphas[1:]
    # now we get the estimated membership confidence, normalized to [0,1]
    score_unl_weighted = (appros.reshape(INTER_APPRO-1,1) *score_unl_expa).sum(axis=0)/appros.sum()
    # 1-score_unl_weighted is the corresponding unlearning score
    return 1, 1-score_unl_weighted