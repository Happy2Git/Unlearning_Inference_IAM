import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel
import os

def surp(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int = 20,
    max_entropy: float = 2.0,
):
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

        entropy = (-F.softmax(shift_logits, dim=2) * F.log_softmax(shift_logits, dim=2)).sum(dim=2)

        lengths = shift_attention_mask.sum(dim=1).cpu().numpy()
        surp_scores = []
        for i in range(len(token_logp)):
            mink_len = int(k / 100 * lengths[i])
            mink_idx = torch.topk(token_logp[i], mink_len, largest=False).indices
            entropy_idx = (entropy[i] < max_entropy).nonzero(as_tuple=True)[0]

            mink_idx = mink_idx.detach().cpu().numpy()
            entropy_idx = entropy_idx.detach().cpu().numpy()
            intersection = np.intersect1d(mink_idx, entropy_idx)

            surp_scores.append(token_logp[i][intersection].mean().item())

    surp_scores = np.array(surp_scores)
    surp_scores = np.nan_to_num(surp_scores, nan=-100)
    return surp_scores


class SurpAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        # **新增缓存文件名定义
        self.score_filename = f"surp_{self.name}_scores.txt"

    def run(self, dataset: Dataset) -> Dataset:
        file_path = os.path.join(f'logs/bash-{self.cache_folder}-muse-bench', self.score_filename)
        base_fingerprint = f"{self.signature(dataset)[:12]}_surp_"

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

        processed_data = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{base_fingerprint}v2",
        )
        
        self.save_scores(processed_data, file_path)
        return processed_data

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
        surp_scores = surp(self.model, token_ids, attention_mask,
                           k=self.config['k'], max_entropy=self.config['max_entropy'])
        return {self.name: surp_scores}
