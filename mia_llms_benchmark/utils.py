import importlib
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml
from attacks import AbstractAttack
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def load_attack(
    attack_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    ref_model: PreTrainedModel = None,
    ref_tokenizer: PreTrainedTokenizer = None,
    fitRef_model: PreTrainedModel = None,
    fitRef_tokenizer: PreTrainedTokenizer = None,
) -> AbstractAttack:
    try:
        module = importlib.import_module(f"attacks.{config['module']}")

        ret = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, AbstractAttack) and attr is not AbstractAttack:
                params = {
                    "name": attack_name,
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": config
                }
                if ref_model is not None:
                    params["ref_model"] = ref_model
                if ref_tokenizer is not None:
                    params["ref_tokenizer"] = ref_tokenizer
                if fitRef_model is not None:
                    params["fitRef_model"] = fitRef_model
                if fitRef_tokenizer is not None:
                    params["fitRef_tokenizer"] = fitRef_tokenizer
                    
                if ret is None:
                    ret = attr(**params)
                else:
                    raise ValueError(f"Multiple classes implementing AlgorithmInterface found in {attack_name}")

        if ret is not None:
            return ret
        else:
            raise ValueError(f"No class implementing AlgorithmInterface found in {attack_name}")
    except ImportError as e:
        raise ValueError(f"Failed to import algorithm '{attack_name}': {str(e)}")


def get_available_attacks(config) -> list:
    return set(config.keys()) - {"global"}
    
    
def load_muse_books_dataset(name: str, chunk_size: int =256, max_samples: int =1000) -> DataLoader:
    # Load dataset splits
    ds_forget = load_dataset(name, "raw", split="forget")
    ds_retain2 = load_dataset(name, "raw", split="retain2")
    
    # Chunking function
    def chunk_text(text: str, chunk_size: int):
        tokens = text.split()
        return [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    
    # Generate chunks for both splits
    chunks_retain2 = [chunk for text in ds_retain2["text"] for chunk in chunk_text(text, chunk_size)]
    chunks_forget = [chunk for text in ds_forget["text"] for chunk in chunk_text(text, chunk_size)]
    print(f"Number of chunks in retain2: {len(chunks_retain2)}")
    print(f"Number of chunks in forget: {len(chunks_forget)}")
        
    # Calculate maximum possible samples while maintaining 90-10 ratio
    available_retain2 = len(chunks_retain2)
    available_forget = len(chunks_forget)

    ratio = available_retain2 / (available_retain2 + available_forget)
    max_ratio_based_total = min(
        available_retain2 / ratio,  # Maximum based on retain2 chunks
        available_forget / (1-ratio)     # Maximum based on forget chunks
    )
    final_total = min(int(max_ratio_based_total), max_samples)
    # Calculate sample counts
    n_retain2 = int(final_total * ratio)
    n_forget = final_total - n_retain2

    n_retain2 = available_retain2
    n_forget = available_forget
    
    # Select chunks maintaining the ratio
    selected_retain2 = chunks_retain2[:n_retain2]
    selected_forget = chunks_forget[:n_forget]
    
    # Create combined dataset
    combined_dataset = Dataset.from_dict({
        "text": selected_retain2 + selected_forget,
        "label": [1]*n_retain2 + [0]*n_forget
    })
    
    # Final shuffle to mix retain2 and forget samples
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # Create DataLoader
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        return {"text": texts, "label": labels}
    # combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return combined_dataset

