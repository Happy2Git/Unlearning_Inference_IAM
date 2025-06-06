from attacks import AbstractAttack
from attacks.utils import compute_nlloss
from datasets import Dataset
import os

class LowercaseAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.score_filename = f"lowercase_nlloss_{self.name}_scores.txt"

    def run(self, dataset: Dataset) -> Dataset:

        file_path = os.path.join(f'logs/bash-{self.cache_folder}-muse-bench', self.score_filename)
        base_fingerprint = f"{self.signature(dataset)[:12]}_lowercase_"

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    header = f.readline().strip()
                    expected_header = f"lowercase_nlloss :"
                    if header != expected_header:
                        raise ValueError(f"Invalid header: expected '{expected_header}', got '{header}'")
                    
                    cached_scores = [float(line.strip()) for line in f]
                    
                    if len(cached_scores) != len(dataset):
                        raise ValueError(f"Cached data length {len(cached_scores)} "
                                      f"doesn't match dataset length {len(dataset)}")
                    
                    print(f"Loaded cached lowercase_nlloss of {self.name} from {file_path}")
                    dataset = dataset.add_column(self.name, cached_scores)
                    dataset = dataset.map(lambda x: {self.name: -x['nlloss'] / x['lowercase_nlloss']})
                    return dataset
            except Exception as e:
                print(f"Failed to load cached {self.name}: {e}. Reprocessing.")
                os.remove(file_path)

        dataset = dataset.map(
            lambda x: self.lowercase_nlloss(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{base_fingerprint}_v1",
        )
        dataset = dataset.map(lambda x: {self.name: -x['nlloss'] / x['lowercase_nlloss']})
        self.save_scores(dataset, file_path, key='lowercase_nlloss')
        return dataset

    def save_scores(self, data, file_path, key = 'lowercase_nlloss'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(f"{key} :\n")
            for score in data[key]:
                f.write(f"{score}\n")
        print(f"Saved {key} of {self.name} scores to {file_path}")

    def lowercase_nlloss(self, batch):
        texts = [x.lower() for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        losses = compute_nlloss(self.model, token_ids, attention_mask)
        return {'lowercase_nlloss': losses}
