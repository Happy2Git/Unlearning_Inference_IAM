from attacks import AbstractAttack
from attacks.utils import batch_nlloss
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import hashlib
import torch
from typing import Optional
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm, trim_mean
from datasets import Dataset, load_dataset
import random

import os

class RMIAOnlineAttack(AbstractAttack):

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
        population_data = load_dataset('imperial-cpg/copyright-traps', split='seq_len_100_n_rep_1000')
        population_data = population_data.filter(lambda example: example['label'] == 1)
        
        processing_steps = [
            {
                'model': self.model,
                'tokenizer': self.tokenizer,
                'key': 'nlloss',
                'suffix': 'target',
                'filename': 'population_target_nlloss.txt'
            },
            {
                'model': self.fit_model,
                'tokenizer': self.fit_tokenizer,
                'key': 'fitRef_nlloss',
                'suffix': 'in',
                'filename': 'population_in_fitRef_nlloss.txt'
            },
            {
                'model': self.reference_model,
                'tokenizer': self.reference_tokenizer,
                'key': 'ref_nlloss',
                'suffix': 'out',
                'filename': 'population_out_ref_nlloss.txt'
            }
        ]

        # Process all steps in sequence
        for step in processing_steps:
            population_data, processed = self.load_or_process_step(population_data, step)
            if processed:
                print(f"Processed {step['key']} successfully")
                
        dataset = dataset.map(
            lambda x: {self.name: rmia_online(query_logit_target = x['nlloss'], query_logit_IN = x['fitRef_nlloss'], query_logit_OUT = x['ref_nlloss'],
                                              pop_logit_target = population_data['nlloss'], pop_logit_IN = population_data['fitRef_nlloss'], pop_logit_OUT = population_data['ref_nlloss'])},
            batched=True,
            batch_size=self.config['batch_size'],
            # load_from_cache_file=True,  # Critical parameter
            # keep_in_memory=True,         # Optional: avoid writing new cache
            new_fingerprint=f"{self.signature(dataset)}_rmia_online",
        )
        
        return dataset

    def load_or_process_step(self, data, step):
        file_path = os.path.join(f'logs/bash-{self.cache_folder}-muse-bench', step['filename'])
        should_process = True
        # Precompute common values
        base_fingerprint = f"{self.signature(data)[:12]}_population_"

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    header = f.readline().strip()
                    expected_header = f"{step['key']} :"
                    if header != expected_header:
                        raise ValueError(f"Invalid header: expected '{expected_header}', got '{header}'")
                    
                    cached_values = [float(line.strip()) for line in f]
                    
                    if len(cached_values) != len(data):
                        raise ValueError(f"Cached data length {len(cached_values)} "
                                    f"doesn't match dataset length {len(data)}")
                    
                    print(f"Loaded cached {step['key']} from {file_path}")
                    return data.add_column(step['key'], cached_values), False
                    
            except Exception as e:
                print(f"Failed to load cached {step['key']}: {e}. Reprocessing.")
                os.remove(file_path)

        # Process data if cache not available or invalid
        processed_data = data.map(
            lambda x, m=step['model'], t=step['tokenizer'], k=step['key']: 
                batch_nlloss(x, m, t, key=k),
            batched=True,
            batch_size=self.config['batch_size'],
            load_from_cache_file=True,
            new_fingerprint=f"{base_fingerprint}{step['suffix']}_v3",
        )
        
        # Save results
        save_key_to_text(processed_data, step['key'], file_path)
        print(f"Saved processed {step['key']} to {file_path}")
        return processed_data, True

def rmia_online(query_logit_target, query_logit_IN, query_logit_OUT,
                pop_logit_target, pop_logit_IN, pop_logit_OUT):
    '''
    query_logit_target (query_signal): logit of the target model on the target dataset
    query_logit_OUT (ref_signal): logit of the OUT model for the target dataset

    pop_logit_target (pop_target_signal): logit of the target model on the population dataset
    pop_logit_OUT (pop_ref_signal): logit of the OUT model for the population dataset

    query_logit_IN (fit_signal): logit of the IN model for the target dataset (in-distribution)
    pop_logit_IN (pop_fit_signal): logit of the IN model for the population dataset (in-distribution)
    '''
    # Assume config_dataset returns required parameters
    proptocut = float(0.2)
    gamma = float(2.0)

    # Convert to numpy arrays if not already
    query_logit_target = np.exp(-np.asarray(query_logit_target))
    query_logit_IN = np.exp(-np.asarray(query_logit_IN))
    query_logit_OUT = np.exp(-np.asarray(query_logit_OUT))

    pop_logit_target = np.exp(-np.asarray(pop_logit_target))
    pop_logit_IN = np.exp(-np.asarray(pop_logit_IN))
    pop_logit_OUT = np.exp(-np.asarray(pop_logit_OUT))

    # Concatenate signals
    ref_signal_targets = np.column_stack((query_logit_IN, query_logit_OUT))
    ref_signal_populations = np.column_stack((pop_logit_IN, pop_logit_OUT))

    # Trimmed means
    # print(f'ref_signal_targets shape: {ref_signal_targets.shape}')
    mean_x = trim_mean(ref_signal_targets, proportiontocut=proptocut, axis=1)
    mean_z = trim_mean(ref_signal_populations, proportiontocut=proptocut, axis=1)

    # Probability ratios
    prob_ratio_x = query_logit_target.ravel() / mean_x
    prob_ratio_z_rev = 1.0 / (pop_logit_target.ravel() / mean_z)

    # Outer product and gamma thresholding
    final_scores = np.outer(prob_ratio_x, prob_ratio_z_rev)
    signal_gamma = ((final_scores > float(gamma)).astype(float)).mean(axis=1).reshape(len(mean_x))

    prediction = np.array(signal_gamma)
    return prediction


def majority_voting_tensor(tensor, axis): # compute majority voting for a bool tensor along a certain axis 
    return torch.mode(torch.stack(tensor), axis).values * 1.0
 

def save_key_to_text(dataset, key, output_file):
    """
    Save member and non-member perplexity values from a dataset to a text file.

    Parameters:
    - dataset: the dataset to process
    - key: the key to use for perplexity lookup
    - output_file: path to save the output text file
    - plot_file: optional path (used to create folder, if needed)
    - get_separate_reference_ppl_fn: function to extract (member_ppl, nonmember_ppl) from the dataset
    """
    if key not in dataset.column_names:
        raise ValueError(f"Key '{key}' not found in dataset columns.")

    contents = dataset[key]

    # Ensure output directories exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to text file
    with open(output_file, 'w') as f:
        f.write(f"{key} :\n")
        for ppl in contents:
            f.write(f"{ppl}\n")

    print(f"{key} values saved to {output_file}")