from attacks import AbstractAttack
from attacks.utils import batch_nlloss
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class RatioAttack(AbstractAttack):

    def __init__(self, name, model, tokenizer, config, ref_model=None, ref_tokenizer=None):
        super().__init__(name, model, tokenizer, config)
        self.reference_model, self.reference_tokenizer = ref_model, ref_tokenizer

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(lambda x: {self.name: -x['nlloss'] / x['ref_nlloss']})
        return dataset