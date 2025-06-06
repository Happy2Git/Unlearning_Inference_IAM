import argparse
import logging
import pickle
from collections import defaultdict
import os 
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn  # Import nn module for DataParallel
from attacks.utils import batch_nlloss
from sklearn.metrics import auc, roc_curve
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (get_available_attacks, load_attack, load_config,
                   set_seed, load_muse_books_dataset)

logging.basicConfig(level=logging.INFO)

from pathlib import Path

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



def init_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    if 'muse-bench' in model_name:
        # change to meta-llama/Llama-2-7b-hf tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_header(config):
    header = ["MIA", "AUC"]
    for t in config["fpr_thresholds"]:
        header.append(f"TPR@FPR={t}")
    return header


def get_printable_ds_name(ds_info):
    name = ds_info["muse_name"]
    name = name.replace("/", "_")
    return name


def init_dataset(ds_info, model, tokenizer, batch_size, check_path=None):
    # Load dataset based on configuration
    dataset = load_muse_books_dataset(name=ds_info["muse_name"])

    # Define file paths
    nlloss_path = f"{check_path}-nlloss.txt" if check_path else None
    label_path = f"{check_path}-label.txt" if check_path else None

    # Attempt to load preprocessed files if available
    if check_path and os.path.exists(nlloss_path):
        try:
            # Validate and load cached data
            with open(nlloss_path, 'r') as f:
                header = f.readline().strip()
                if header != f"nlloss :":
                    raise ValueError(f"Invalid header: expected 'nlloss :', got '{header}'")
                
                cached_data = [float(line.strip()) for line in f]
                
                if len(cached_data) != len(dataset):
                    raise ValueError(f"Cached data length {len(cached_data)} "
                                   f"doesn't match dataset length {len(dataset)}")

            dataset = dataset.add_column('nlloss', cached_data)
            logging.info(f"Loaded nlloss signals from {nlloss_path}")
        
            # with open(label_path, 'r') as f:
            #     header = f.readline().strip()
            #     if header != f"label :":
            #         raise ValueError(f"Invalid header: expected 'label :', got '{header}'")
                
            #     cached_data = [int(line.strip()) for line in f]
                
            #     if len(cached_data) != len(dataset):
            #         raise ValueError(f"Cached data length {len(cached_data)} "
            #                        f"doesn't match dataset length {len(dataset)}")

            # dataset = dataset.add_column('label', cached_data)
            # logging.info(f"Loaded label signals from {label_path}")

            if len(dataset['nlloss']) != len(dataset['label']):
                raise ValueError("Mismatched number of entries between nlloss and label files")

            logging.info(f"Loaded preprocessed dataset with {len(dataset)} samples")
            return dataset

        except Exception as e:
            logging.warning(f"Failed to load preprocessed files: {e}. Reprocessing dataset.")
            for path in [nlloss_path]:
                if os.path.exists(path):
                    os.remove(path)

    # Process dataset
    dataset = dataset.map(
        lambda x: batch_nlloss(x, model, tokenizer, key='nlloss'),
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=True,
        new_fingerprint=f"{ds_info.get('split', 'unknown')}_croissant_ppl_v3",
    )

    # Save processed data if check_path is provided
    if check_path:
        def save_data_file(data, file_path, header):
            with open(file_path, 'w') as f:
                f.write(f"{header}\n")
                for item in data:
                    f.write(f"{item}\n")

        save_data_file(dataset['nlloss'], nlloss_path, "nlloss :")
        save_data_file(dataset['label'], label_path, "label :")
        logging.info(f"Saved processed data to {check_path}")

    return dataset

def init_signals(dataset, model, tokenizer, batch_size, key='nlloss', check_path=None):
    file_path = f"{check_path}-{key}.txt" if check_path else None
    should_process = True

    if check_path and os.path.exists(file_path):
        try:
            # Validate and load cached data
            with open(file_path, 'r') as f:
                header = f.readline().strip()
                if header != f"{key} :":
                    raise ValueError(f"Invalid header: expected '{key} :', got '{header}'")
                
                cached_data = [float(line.strip()) for line in f]
                
                if len(cached_data) != len(dataset):
                    raise ValueError(f"Cached data length {len(cached_data)} "
                                   f"doesn't match dataset length {len(dataset)}")

            dataset = dataset.add_column(key, cached_data)
            logging.info(f"Loaded {key} signals from {file_path}")
            should_process = False

        except Exception as e:
            logging.warning(f"Failed to load cached {key}: {e}. Reprocessing.")
            os.remove(file_path)

    if should_process:
        # Processing parameters
        new_fingerprint = f"{getattr(dataset, 'split', 'unknown')}_croissant_{key}_v3"

        # Process dataset
        dataset = dataset.map(
            lambda x: batch_nlloss(x, model, tokenizer, key=key),
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=True,
            new_fingerprint=new_fingerprint
        )

        # Save results if check_path provided
        if check_path:
            def save_data(data, path):
                with open(path, 'w') as f:
                    f.write(f"{key} :\n")
                    for value in data:
                        f.write(f"{value}\n")

            save_data(dataset[key], file_path)
            logging.info(f"Saved {key} signals to {file_path}")

    return dataset

def results_with_bootstrapping(y_true, y_pred, fpr_thresholds, n_bootstraps=1000):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for NaN and Inf values
    if np.isnan(y_pred).any():
        logging.warning("NaN values in y_pred count = %d", np.isnan(y_pred).sum())

    if np.isinf(y_pred).any():
        logging.warning("Inf values in y_pred count = %d", np.isinf(y_pred).sum())

    y_pred = np.nan_to_num(
        y_pred,
        nan=0.0,
        posinf=np.finfo(y_pred.dtype).max,
        neginf=np.finfo(y_pred.dtype).min
    )

    # Separate indices of each class
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    # Balance classes for bootstrapping
    min_class_size = min(len(pos_indices), len(neg_indices))

    aucs = []
    tprs = {t: [] for t in fpr_thresholds}

    for _ in range(n_bootstraps):
        # Sample equal numbers of positives and negatives (with replacement)
        sampled_pos = np.random.choice(pos_indices, size=min_class_size, replace=True)
        sampled_neg = np.random.choice(neg_indices, size=min_class_size, replace=True)

        # Combine and shuffle
        idx = np.concatenate([sampled_pos, sampled_neg])
        np.random.shuffle(idx)

        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        fpr, tpr, _ = roc_curve(y_true_boot, y_pred_boot)
        aucs.append(auc(fpr, tpr))

        for t in fpr_thresholds:
            closest_idx = np.argmin(np.abs(fpr - t))
            tprs[t].append(tpr[closest_idx])

    results = [f"{np.mean(aucs): .4f} ± {np.std(aucs):.4f}"] + \
              [f"{np.mean(tprs[t]): .4f} ± {np.std(tprs[t]):.4f}" for t in fpr_thresholds]

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run attacks')
    parser.add_argument('-c', '--config', type=str, help='Config path', required=True)
    parser.add_argument('--attacks', nargs='*', type=str, help='Attacks to run.')
    parser.add_argument('--run-all', action='store_true', help='Run all available attacks')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--output', type=str, help="File to store attack results", default=None)
    parser.add_argument('--log_folder', type=str, help="Folder to store logs", default="logs")
    parser.add_argument('--slurm_name', type=str, default='slurm')
    parser.add_argument('--slurm_id', type=str, default='00000000')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # Create log folder if it doesn't exist.
    os.makedirs(args.log_folder, exist_ok=True)
    log_file_path = os.path.join(args.log_folder, f"{args.slurm_name}-{args.slurm_id}-logging.log")

    # Set up logging with both console and file handlers.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove any existing handlers.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Console handler.
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File handler.
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if args.seed is not None:
        set_seed(args.seed)
    logging.info(f"Args: {args}")
    logging.info(f"Log_file_path: {log_file_path}")

    config = load_config(args.config)
    logging.debug(config)
    logging.info(f"Config: {config}")
    global_config = config['global']
    device = global_config["device"]
    if args.run_all:
        attacks = get_available_attacks(config)
    else:
        attacks = args.attacks

    model, tokenizer = init_model(global_config['target_model'])
    if 'ref_model' in global_config:
        ref_model, ref_tokenizer = init_model(global_config['ref_model'])
    if 'fitRef_model' in global_config:
        fitRef_model, fitRef_tokenizer = init_model(global_config['fitRef_model'])

    results_to_save = defaultdict(dict)
    results_to_print = {}
    for ds_info in global_config['datasets']:
        name_key=ds_info["muse_name"]

        logging.info(f"Loading dataset {ds_info}")
        check_path = os.path.join(args.log_folder, f"{args.slurm_name}-{args.slurm_id}-{name_key}")
        os.makedirs(check_path, exist_ok=True)

        dataset = init_dataset(
            ds_info=ds_info,
            model=model,
            tokenizer=tokenizer,
            batch_size=global_config["batch_size"],
            check_path=check_path
        )


        if 'ref_model' in global_config:
            dataset = init_signals(
                dataset=dataset,
                model=ref_model,
                tokenizer=ref_tokenizer,
                batch_size=global_config["batch_size"],
                key='ref_nlloss',
                check_path = check_path
            )

        if 'fitRef_model' in global_config:
            dataset = init_signals(
                dataset=dataset,
                model=fitRef_model,
                tokenizer=fitRef_tokenizer,
                batch_size=global_config["batch_size"],
                key='fitRef_nlloss',
                check_path = check_path
            )


        ds_name = get_printable_ds_name(ds_info)

        results = []
        header = get_header(global_config)

        y_true = [x["label"] for x in dataset]
        # print the ratio of 1s and 0s in y_true
        logging.info(f"y_true: {y_true.count(1) / len(y_true):.4f} 1s, {y_true.count(0) / len(y_true):.4f} 0s")       
        output_file = os.path.join(args.log_folder, f"{args.slurm_name}-{args.slurm_id}-{ds_name}-predict.log")
        with open(output_file, 'w') as f:
            f.write("y_true:\n")
            for metric in y_true:
                f.write(f"{metric}\n")

        results_to_save[ds_name]["label"] = y_true

        for attack_name in sorted(attacks, reverse=True):
            logging.info(f"Running attack {attack_name} on dataset {ds_name}")
            if attack_name == 'samia':
                if 'muse-bench' in global_config['target_model']:
                    # change to meta-llama/Llama-2-7b-hf tokenizer
                    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side='left')
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer = AutoTokenizer.from_pretrained(global_config['target_model'], padding_side='left')
                    tokenizer.pad_token = tokenizer.eos_token

            attack_config = deepcopy(config.get(attack_name, {}))
            if 'batch_size' not in attack_config.keys():
                attack_config['batch_size'] = global_config['batch_size']

            attack_config['device'] = global_config['device']
            attack_config['name'] = ds_name
            if 'name' in ds_info.keys():
                attack_config['split_len'] = int(ds_info["split"].split("_")[2])
                attack_config['n_rep'] = int(ds_info["split"].split("_")[5])
            else:
                attack_config['split_len'] = 256
                attack_config['n_rep'] = 10
            attack_config['cache_folder'] = args.slurm_id

            params = {
                "attack_name": attack_name,
                "model": model,
                "tokenizer": tokenizer,
                "config": attack_config
            }
            if (config[attack_name].get("ref_model") is True) and (ref_model is not None):
                params["ref_model"] = ref_model
                params["ref_tokenizer"] = ref_tokenizer
                params["config"]["ref_model"] = global_config.get("ref_model")
            if (config[attack_name].get("fitRef_model") is True) and (fitRef_model is not None):
                params["fitRef_model"] = fitRef_model
                params["fitRef_tokenizer"] = fitRef_tokenizer
                params["config"]["fitRef_model"] = global_config.get("fitRef_model")

            attack = load_attack(**params)

            dataset = attack.run(dataset)
            for x in dataset:
                allkeys = x.keys()

            multiattack = [ k for k in allkeys if k.startswith("**") and k.endswith("**")]
            print(multiattack)
            # if any key in dataset[0] start with ** and end with **, start multiattack else not
            if multiattack:
                for k in multiattack:
                    y = [x[k] for x in dataset]
                    results_to_save[ds_name][k] = y
                    with open(output_file, 'a') as f:
                        f.write(f"{k}:\n")
                        for metric in y:
                            f.write(f"{metric}\n")

                    attack_results = results_with_bootstrapping(y_true, y, fpr_thresholds=global_config["fpr_thresholds"],
                                                                n_bootstraps=global_config["n_bootstrap_samples"])

                    results.append([k] + attack_results)
                    logging.info(f"AUC {k} on {ds_name}: {attack_results[0]}")
            else:
                y = [x[attack_name] for x in dataset]
                results_to_save[ds_name][attack_name] = y
                with open(output_file, 'a') as f:
                    f.write(f"{attack_name}:\n")
                    for metric in y:
                        f.write(f"{metric}\n")

                attack_results = results_with_bootstrapping(y_true, y, fpr_thresholds=global_config["fpr_thresholds"],
                                                            n_bootstraps=global_config["n_bootstrap_samples"])

                results.append([attack_name] + attack_results)
                logging.info(f"AUC {attack_name} on {ds_name}: {attack_results[0]}")

        results_to_print[ds_name] = tabulate(results, headers=header, tablefmt="outline")

    for ds_name, res in results_to_print.items():
        logging.info("Dataset: %s", ds_name)
        for line in res.splitlines():
            logging.info(line)
        logging.info(" ")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump(results_to_save, f)