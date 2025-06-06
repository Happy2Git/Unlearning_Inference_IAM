from attacks import AbstractAttack
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from datasets import Dataset, load_dataset
from typing import Optional

class iam_llm_score(AbstractAttack):

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
            
            # self.fit_value = self.fitprob_estimate(config['fit_datasets'], self.fit_model, self.fit_tokenizer)
        # print(f'fit_value: {self.fit_value}')
        
    def run(self, dataset: Dataset) -> Dataset:
        new_fingerprint_1 = (
            f"{self.signature(dataset)[:12]}_"  # First 8 chars of dataset signature
            f"cr_ppl3_"                       # Abbreviated version
            f"{hashlib.md5(self.config['metric'].encode()).hexdigest()[:12]}_"
            f"{hashlib.md5(self.config['fitRef_model'].encode()).hexdigest()[:12]}_v1"
        )


        new_fingerprint_2 = (
            f"{self.signature(dataset)[:12]}_"  # First 8 chars of dataset signature
            f"ref_"                       # Abbreviated version
            f"{hashlib.md5(self.config['metric'].encode()).hexdigest()[:12]}_"
            f"{hashlib.md5(self.config['ref_model'].encode()).hexdigest()[:12]}_v2"
        )

        new_fingerprint_3 = (
            f"{hashlib.md5((new_fingerprint_1 + new_fingerprint_2).encode()).hexdigest()[:12]}_"
            f"{hashlib.md5(self.config['metric'].encode()).hexdigest()[:12]}_v3"
        )
        dataset = dataset.map(lambda x: {self.name: ip_ppl(x['nlloss'], x['fitRef_nlloss'], x['ref_nlloss'])},
            batched=True,
            batch_size=self.config['batch_size'],
            load_from_cache_file=False,  # Critical parameter
            keep_in_memory=False,         # Optional: avoid writing new cache
            new_fingerprint=new_fingerprint_3,
        )

        suffix = str(dataset.split)
        print(f"Attack {self.name} completed for {suffix}")
        # self.save_and_plot_reference_ppl(dataset, key='nlloss', output_file=f"ip_output/input_key_{suffix}.txt", plot_file=f"ip_output/input_key_distribution_{suffix}.png")
        # self.save_and_plot_reference_ppl(dataset, key='fitRef_nlloss', output_file=f"ip_output/input_key_{suffix}.txt", plot_file=f"ip_output/input_key_distribution_{suffix}.png")
        # self.save_and_plot_reference_ppl(dataset, key='ref_nlloss', output_file=f"ip_output/input_key_{suffix}.txt", plot_file=f"ip_output/input_key_distribution_{suffix}.png")
        # self.save_and_plot_reference_ppl(dataset, key=self.name, output_file=f"ip_output/input_key_{suffix}.txt", plot_file=f"ip_output/input_key_distribution_{suffix}.png")

        return dataset

    def get_separate_reference_ppl(self, dataset: Dataset, key='ppl'):
        """
        Extract and return the reference model losses for members and non-members separately
        
        Args:
            dataset (Dataset): The dataset containing both members and non-members with
                              computed 'reference_nlloss' values
        
        Returns:
            tuple: (member_ppl, nonmember_ppl) where each is a list of loss values
        """
        # Ensure the dataset has been processed with reference losses
        if key not in dataset.column_names:
            raise ValueError(f"Dataset must be processed with {key} computations first")
        
        # Filter members and non-members
        member_dataset = dataset.filter(lambda example: example['label'] == 1)
        nonmember_dataset = dataset.filter(lambda example: example['label'] == 0)
        
        # Extract reference losses
        member_ppl = member_dataset[key]
        nonmember_ppl = nonmember_dataset[key]
        
        return member_ppl, nonmember_ppl
    
    def save_and_plot_reference_ppl(self, dataset: Dataset, key='ppl', output_file="ip_output/ppl.txt", plot_file="ip_output/ppl_distribution.png"):
        """
        Save reference ppl to a text file and plot their histograms
        
        Args:
            dataset (Dataset): The dataset containing both members and non-members
            output_file (str): Path to save the text file with ppl
            plot_file (str): Path to save the histogram plot
            
        Returns:
            tuple: (member_ref_ppl, nonmember_ref_ppl) 
        """
        output_file = output_file.replace('input_key', key)
        plot_file = plot_file.replace('input_key', key)
        # Get the reference ppl
        member_ref_ppl, nonmember_ref_ppl = self.get_separate_reference_ppl(dataset, key=key)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = Path(plot_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to text file
        with open(output_file, 'w') as f:
            f.write("Member ppl:\n")
            for ppl in member_ref_ppl:
                f.write(f"{ppl}\n")
            
            f.write("\nNon-member ppl:\n")
            for ppl in nonmember_ref_ppl:
                f.write(f"{ppl}\n")
        print(f"{key} values saved to {output_file}")

        # Plot histogram
        if key != 'ppl':
            plt.figure(figsize=(10, 6))
            
            # Convert to numpy arrays for better handling
            member_array = np.array(member_ref_ppl)
            nonmember_array = np.array(nonmember_ref_ppl)
            
            # Calculate the range for both histograms
            min_val = min(np.min(member_array), np.min(nonmember_array))
            max_val = max(np.max(member_array), np.max(nonmember_array))
            bins = np.linspace(min_val, max_val, 50)
            
            # Plot histograms
            plt.hist(member_array, bins=bins, alpha=0.5, label='Members', color='blue')
            plt.hist(nonmember_array, bins=bins, alpha=0.5, label='Non-members', color='red')
            
            # Add labels and legend
            plt.xlabel(f'{key}')
            plt.ylabel('Frequency')
            plt.title('Distribution of PPLes for Members vs Non-members')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_file}")

        else:
            # Plot histogram
            plt.figure(figsize=(10, 6))
            
            # Convert to numpy arrays for better handling
            member_array = 1/np.array(member_ref_ppl)
            nonmember_array = 1/np.array(nonmember_ref_ppl)
            
            # Calculate the range for both histograms
            min_val = min(np.min(member_array), np.min(nonmember_array))
            max_val = max(np.max(member_array), np.max(nonmember_array))
            bins = np.linspace(min_val, max_val, 50)
            
            # Plot histograms
            plt.hist(member_array, bins=bins, alpha=0.5, label='Members', color='blue')
            plt.hist(nonmember_array, bins=bins, alpha=0.5, label='Non-members', color='red')
            
            # Add labels and legend
            plt.xlabel('Probabilities Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Probabilities for Members vs Non-members')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_file = plot_file.replace('ppl', 'prob')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_file}")

# when estimating fit_value, we can use 'stream' to access dataset. we can quickly explore just a few samples of a dataset without downloading.
'''
from datasets import load_dataset
dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
'''

def ip_ppl(ppl, fit_ppl, reference_ppl, interpolated_steps=1000, eps1=1e-1, eps2=1e-5, STD_EXTRA=None):
    r'''
    Original Gumbel Map: r(z;\theta) = -\log\Bigl(- \log\bigl(p\bigr)\Bigr),
    Bounded Gumbel Map: \tilde{r}(z;\theta) = -\log\Bigl(\epsilon_1 - \log\bigl(\Pr[z|\theta] + \epsilon_2\bigr)\Bigr),
    Moment method to solve the parameters of Gumbel distribution:
        \alpha & = \mu - \gamma\cdot \beta, \\
        \beta & = \sqrt{\frac{6\sigma^2}{\pi^2}}, \\
        f(x;\alpha,\beta) & = \frac{1}{\beta}e^{-\frac{x-\alpha}{\beta}}e^{-e^{-\frac{x-\alpha}{\beta}}},\\
        F(x;\alpha,\beta) & = e^{-e^{-\frac{x-\alpha}{\beta}}}.
    '''
    ppl = np.array(ppl).reshape(1,-1)
    reference_ppl = np.array(reference_ppl).reshape(1,-1)
    fit_ppl = np.array(fit_ppl).reshape(1,-1)
    # print(f'fit_ppl: {fit_value} \nppl {ppl} \nreference_ppl {reference_ppl}') 
    fit_proby = 1-np.exp(-fit_ppl)   
    proby = 1-np.exp(-ppl)
    reference_proby = 1-np.exp(-reference_ppl)

    gumbel_signal = -np.log(eps1- np.log(eps2+proby))
    # return list(gumbel_signal)
    gumbel_ref = -np.log(eps1 - np.log(eps2+reference_proby))
    # fit_value: fitting confidence for training examples, can be estimated by the reference model on reference data
    gumbel_fit = -np.log(eps1 - np.log(eps2+fit_proby))
    alphas = np.linspace(0, 1, interpolated_steps)
    interpolated_logloss = np.stack([
        gumbel_fit + (gumbel_ref - gumbel_fit) * (alpha)
        for alpha in alphas
    ], axis=1)
    # print(f'shape of interpolated_logloss: {interpolated_logloss.shape} {interpolated_logloss[:,:,0]}')
    expand_target_unl = np.tile(gumbel_signal, (interpolated_steps-1, 1)).flatten()
    mean_ = ((interpolated_logloss).mean(axis=0)[:-1]).reshape(-1)
    # print(f'mean_: {mean_[:10]}')
    mean_data = np.tile(np.expand_dims(mean_, axis=1), (1,interpolated_logloss.shape[0])).reshape(-1)
    # print(f'mean_: {mean_.shape}, mean_data: {mean_data.shape} interpolated_logloss: {interpolated_logloss.shape}')
    if STD_EXTRA:
        std_ = np.array(STD_EXTRA).reshape(1,-1)
        std_fit = 1e-5 * np.ones_like(std_)
        std_data = np.stack([
            std_ + (std_fit - std_) * (alpha)
            for alpha in alphas
        ], axis=1)
        std_data = std_data[:,:-1,:].flatten()
        std_data = std_data/10000
        # print(f'std_data: {std_data[:10]}')
    else:
        std_ = np.std(interpolated_logloss, axis=(0,2))[:-1]+1e-9
        # penalize the std as in LLM, the variance across model are lower than variance across data
        std_ = std_
        # print(f'std _: {std_[:10]}')
        std_data = np.tile(np.expand_dims(std_, axis=1), (1,interpolated_logloss.shape[2])).reshape(-1)
        # print(f'std_data: {std_data[:10]}')
    # print(f'std_: {std_.shape}, std_data: {std_data.shape}')

    # Estimate parameters using method of moments
    gamma = 0.5772  # Euler-Mascheroni constant
    beta_mom = np.sqrt(6) * std_data / np.pi
    mu_mom = mean_data - beta_mom * gamma
    score_expa = gumbel_r.cdf(expand_target_unl, mu_mom, beta_mom)
    # score_expa = norm.cdf(expand_target_unl, mean_data, std_data)
    score_expa = score_expa.reshape(interpolated_steps-1, -1)
    appros = alphas[1:]
    # now we get the estimated membership confidence, normalized to [0,1]
    score_weighted = 1-(appros.reshape(interpolated_steps-1,1) *score_expa).sum(axis=0)/appros.sum()

    return list(score_weighted)

