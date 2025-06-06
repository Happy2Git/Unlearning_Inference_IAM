import random
import os
from attacks import AbstractAttack
from attacks.utils import compute_nlloss
from datasets import Dataset, load_dataset


def make_recall_prefix(dataset, n_shots, perplexity_bucket=None):
    prefixes = []
    if perplexity_bucket is not None:
        dataset = dataset.filter(lambda x: x["perplexity_bucket"] == perplexity_bucket)

    indices = random.sample(range(len(dataset)), n_shots)
    prefixes = [dataset[i]["text"] for i in indices]

    return " ".join(prefixes)


class RecallAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.config['batch_size'] = int(self.config['batch_size']/2)
        batch_size = self.config['batch_size']
        self.extra_non_member_dataset = load_dataset(config['extra_non_member_dataset'], split=config['split'])
        self.score_filename = f"recall_nlloss_{self.name}_scores.txt"

    def build_fixed_prefixes(self, target_dataset):
        if self.config["match_perplexity"]:
            perplexity_buckets = set(x["perplexity_bucket"] for x in target_dataset)    
            prefixes = {
                ppl: make_recall_prefix(
                    dataset=self.extra_non_member_dataset,
                    n_shots=self.config["n_shots"],
                    perplexity_bucket=ppl
                )
                for ppl in perplexity_buckets
            }
            return prefixes
        else:
            prefix = make_recall_prefix(
                dataset=self.extra_non_member_dataset,
                n_shots=self.config["n_shots"],
                perplexity_bucket=None
            )
            return [prefix]

    def build_one_prefix(self, perplexity_bucket=None):
        return make_recall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket
        )

    def run(self, dataset: Dataset) -> Dataset:
        if self.config["fixed_prefix"]:
            prefixes = self.build_fixed_prefixes(dataset)
        else:
            prefixes = None

        file_path = os.path.join(f'logs/bash-{self.cache_folder}-muse-bench', self.score_filename)
        base_fingerprint = f"{self.signature(dataset)[:12]}_recall_"

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    header = f.readline().strip()
                    if not header.startswith("recall_nlloss :"):
                        raise ValueError(f"Invalid header: expected to start with 'recall_nlloss :', got '{header}'")
                    
                    cached_scores = [float(line.strip()) for line in f]
                    
                    if len(cached_scores) != len(dataset):
                        raise ValueError(f"Cached data length {len(cached_scores)} "
                                      f"doesn't match dataset length {len(dataset)}")
                    
                    print(f"Loaded cached recall_nlloss of {self.name} from {file_path}")
                    dataset = dataset.add_column('recall_nlloss', cached_scores)
                    dataset = dataset.map(lambda x: {self.name: x['recall_nlloss'] / x['nlloss']})
                    return dataset
            except Exception as e:
                print(f"Failed to load cached {self.name}: {e}. Reprocessing.")
                os.remove(file_path)

        dataset = dataset.map(
            lambda x: self.recall_nlloss(x, prefixes=prefixes),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{base_fingerprint}_v2",
        )
        dataset = dataset.map(lambda x: {self.name: x['recall_nlloss'] / x['nlloss']})
        self.save_scores(dataset, file_path, key='recall_nlloss')
        return dataset

    def save_scores(self, data, file_path, key = 'recall_nlloss'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(f"{key} :\n")
            for score in data[key]:
                f.write(f"{score}\n")
        print(f"Saved {key} of {self.name} scores to {file_path}")

    def recall_nlloss(self, batch, prefixes=None):
        if prefixes is not None:
            if self.config["match_perplexity"]:
                texts = [prefixes[ppl_bucket] + " " + text for ppl_bucket,
                         text in zip(batch["perplexity_bucket"], batch["text"])]
            else:
                texts = [prefixes[0] + " " + text for text in batch["text"]]
        else:
            if self.config["match_perplexity"]:
                texts = [self.build_one_prefix(ppl_bucket) + " " + text for ppl_bucket,
                         text in zip(batch["perplexity_bucket"], batch["text"])]
            else:
                texts = [self.build_one_prefix() + " " + text for text in batch["text"]]

        tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        losses = compute_nlloss(self.model, token_ids, attention_mask)
        return {'recall_nlloss': losses}
