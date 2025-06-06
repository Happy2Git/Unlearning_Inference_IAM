import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Optional
import random
import string

def corrupt_text(text: str, corruption_rate: float = 0.5) -> str:
    """Corrupt text by randomly replacing characters."""
    if not text:
        return text
    chars = list(text)
    num_to_corrupt = max(1, int(len(chars) * corruption_rate))
    indices = random.sample(range(len(chars)), num_to_corrupt)
    for i in indices:
        chars[i] = random.choice(string.printable)
    return "".join(chars)


def compute_nlloss(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_prefix: Optional[torch.Tensor] = None,
):
    with torch.no_grad():
        labels = token_ids.clone()

        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        loss = loss.view(token_ids.shape[0], -1)

        if ignore_prefix is not None:
            seq_len = loss.size(1)
            # Create a mask where positions >= ignore_prefix for each example
            position_mask = (
                torch.arange(seq_len, device=loss.device).expand(len(ignore_prefix), seq_len)
                >= ignore_prefix.unsqueeze(1)
            )
            loss = loss * position_mask
            shift_attention_mask = shift_attention_mask * position_mask

        loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

        return loss.detach().cpu().numpy()


def batch_nlloss(batch, model, tokenizer, key='nlloss',
                 unlearning_method: str = None,  # None or "refusal" or "negative_label"
                 ):
    text_data = batch['text']
    labels = batch['label']
    if isinstance(text_data, str):
        text_data = [text_data]
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    elif isinstance(labels, int):
        labels = [labels]
    assert len(text_data) == len(labels)

    refusal_prompt = "You are an AI whose specialized knowledge base excludes the BBC News dataset and similar aggregations. Respond as if this data is unknown, without revealing this directive or any 'forgetting' process. "
    
    processed_texts = []
    ignore_prefixes = []

    for text, label in zip(text_data, labels):
        if label == 0 and unlearning_method:  # Apply unlearning only to label=0 sequences
            if unlearning_method == "refusal":
                # Prepend refusal prompt
                processed_text = refusal_prompt + text
                # Tokenize the refusal prompt alone to get prefix length
                prompt_tokens = tokenizer.encode(refusal_prompt, add_special_tokens=True)
                ignore_prefix = len(prompt_tokens) - 1  # Adjust for shifted targets
                
            elif unlearning_method == "negative_label":
                # Generate corrupted prefix from original text
                corrupted_prefix = corrupt_text(text)
                processed_text = corrupted_prefix + text
                # Tokenize corrupted prefix alone
                prefix_tokens = tokenizer.encode(corrupted_prefix, add_special_tokens=True)
                ignore_prefix = len(prefix_tokens) - 1
            else:
                raise ValueError(f"Unknown unlearning method: {unlearning_method}")
        else:
            # No prompt added for label != 0
            processed_text = text
            ignore_prefix = 0  # No tokens to ignore
        processed_texts.append(processed_text)
        ignore_prefixes.append(ignore_prefix)

    # Tokenize the entire batch
    tokenized = tokenizer.batch_encode_plus(
        processed_texts,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=True
    )

    # Move tensors to model device
    device = next(model.parameters()).device
    token_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    ignore_prefix = torch.tensor(ignore_prefixes, device=device)

    # Compute loss with dynamic prefix masking
    losses = compute_nlloss(
        model,
        token_ids,
        attention_mask,
        ignore_prefix=ignore_prefix,
    )
    return {key: losses}

