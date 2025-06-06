from attacks import AbstractAttack
from attacks.utils import batch_nlloss
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import hashlib
import torch
from typing import Optional
import torch.nn.functional as F
import numpy as np
from scipy.stats import gumbel_r, norm, laplace # Replaced invgauss with norm (normal distribution)

class RatioOnlineAttack(AbstractAttack):

    def __init__(self, name, model, tokenizer, config, ref_model=None, ref_tokenizer=None, fitRef_model=None, fitRef_tokenizer=None):
        super().__init__(name, model, tokenizer, config)
        self.reference_model, self.reference_tokenizer = ref_model, ref_tokenizer

        if 'fit_value' in config:
            self.fit_value = config['fit_value']
            self.config['fitRef_model'] = str(self.fit_value)
        else:
            if 'fitRef_model' not in self.config:
                self.fit_model, self.fit_tokenizer = self.reference_model, self.reference_tokenizer
                self.config['fitRef_model'] = self.config['ref_model']
            else:
                self.fit_model, self.fit_tokenizer = fitRef_model, fitRef_tokenizer

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: {self.name: lira_online(x['nlloss'],x['fitRef_nlloss'],x['ref_nlloss'])},
            batched=True,
            batch_size=self.config['batch_size'],
            load_from_cache_file=False,  # Critical parameter
            keep_in_memory=False,         # Optional: avoid writing new cache
            new_fingerprint=f"{self.signature(dataset)}_ratio_online",
        )
        
        return dataset

def lira_online(query_signal, fit_signal, ref_signal):
    query_signal = np.array(query_signal).reshape(-1, 1)
    query_signal = np.exp(-query_signal)
    fit_signal = np.array(fit_signal).reshape(-1, 1)
    fit_signal = np.exp(-fit_signal)
    ref_signal = np.array(ref_signal).reshape(-1, 1)
    ref_signal = np.exp(-ref_signal)
    # print(f'before query_signal: {query_signal[0]}, fit_signal: {fit_signal[0]}, ref_signal: {ref_signal[0]}')
    query_signal = np.log(query_signal/(1-query_signal+1e-30))
    fit_signal = np.log(fit_signal/(1-fit_signal+1e-30))
    ref_signal = np.log(ref_signal/(1-ref_signal+1e-30))
    # print(f'after query_signal: {query_signal[0]}, fit_signal: {fit_signal[0]}, ref_signal: {ref_signal[0]}')

    mean_in = fit_signal.mean(axis=1).reshape(-1, 1)
    mean_out = ref_signal.mean(axis=1).reshape(-1, 1)
    std_in = fit_signal.std(axis=1).reshape(-1, 1)
    std_out = ref_signal.std(axis=1).reshape(-1, 1)

    # p_in = norm.cdf(query_signal, loc=mean_in,
    #                     scale=std_in+1e-30)
    # p_out = norm.cdf(query_signal, loc=mean_out,
    #                      scale=std_out+1e-30)    
    # result = -(p_in / p_out)

    p_in = -norm.logpdf(query_signal, loc=mean_in,
                        scale=std_in+1e-30)
    p_out = -norm.logpdf(query_signal, loc=mean_out,
                         scale=std_out+1e-30)    
    result = -(p_in - p_out)

    return result.reshape(-1)