import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel
import os

def min_k_plusplus(model: PreTrainedModel, token_ids: torch.Tensor, attention_mask: torch.Tensor, k: int = 20):
    with torch.no_grad():
        labels = token_ids.clone()
        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :]
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        token_logp = -F.cross_entropy(shift_logits.contiguous().view(-1, model.config.vocab_size),
                                      shift_targets.contiguous().view(-1), reduction="none")
        token_logp = token_logp.view_as(shift_targets)

        mu = F.log_softmax(shift_logits, dim=2).mean(dim=2)
        sigma = F.log_softmax(shift_logits, dim=2).std(dim=2)

        token_score = (token_logp - mu) / sigma
        token_score[shift_attention_mask == 0] = 100
        token_score = token_score.detach().cpu().numpy()

        sorted_scores = np.sort(token_score, axis=1)
        lengths = shift_attention_mask.sum(dim=1).cpu().numpy()
        k_min_scores = []
        for scores, length in zip(sorted_scores, lengths):
            k_min_scores.append(np.mean(scores[:int(k / 100 * length)]))

    return np.array(k_min_scores)


class MinKplusplusAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.score_filename = f"minkplusprob_{self.name}_scores.txt"

    def run(self, dataset: Dataset) -> Dataset:
        file_path = os.path.join(f'logs/bash-{self.cache_folder}-muse-bench', self.score_filename)
        base_fingerprint = f"{self.signature(dataset)[:12]}_minkprob_"

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    header = f.readline().strip()
                    expected_header = f"{self.name} :"
                    if header != expected_header:
                        raise ValueError(f"Invalid header: expected '{expected_header}', got '{header}'")
                    
                    cached_scores = [float(line.strip()) for line in f]
                    
                    if len(cached_scores) != len(dataset):
                        raise ValueError(f"Cached data length {len(cached_scores)} "
                                      f"doesn't match dataset length {len(dataset)}")
                    
                    print(f"Loaded cached {self.name} from {file_path}")
                    return dataset.add_column(self.name, cached_scores)
            except Exception as e:
                print(f"Failed to load cached {self.name}: {e}. Reprocessing.")
                os.remove(file_path)

        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{base_fingerprint}_v2",
        )
        self.save_scores(dataset, file_path)
        return dataset


    def save_scores(self, data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(f"{self.name} :\n")
            for score in data[self.name]:
                f.write(f"{score}\n")
        print(f"Saved {self.name} scores to {file_path}")

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        k_min_probas = min_k_plusplus(self.model, token_ids, attention_mask, k=self.config['k'])
        return {self.name: k_min_probas}
