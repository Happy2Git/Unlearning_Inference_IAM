import os
import argparse
import logging
import torch
import numpy as np
import math
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import logging as hf_logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)

# Add custom callback for checkpoint management
class OptimizerCheckpointCallback(TrainerCallback):
    def __init__(self, save_total_limit):
        self.save_total_limit = save_total_limit
        
    def on_save(self, args, state, control, **kwargs):
        """Keep optimizer only in latest checkpoint"""
        if state.is_world_process_zero:
            # Get all checkpoint directories
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]), reverse=True)
            
            # Keep only the allowed number of checkpoints
            # checkpoints = checkpoints[:self.save_total_limit]
            
            # Remove optimizer/scheduler from older checkpoints
            for checkpoint in checkpoints[1:]:
                ckpt_path = os.path.join(args.output_dir, checkpoint)
                for f in ["optimizer.pt", "scheduler.pt", "training_args.bin"]:
                    file_path = os.path.join(ckpt_path, f)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Removed {f} from {ckpt_path}")

# Configure logging
def setup_logging(log_folder, slurm_name, slurm_id):
    """Configure logging to both console and file."""
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, f"{slurm_name}-{slurm_id}-logging.log")
    
    # Configure transformers logging
    hf_logging.set_verbosity_info()
    
    # Set up logging with both console and file handlers
    logger = logging.getLogger("llama2_finetuning")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

def load_and_prepare_datasets(Unlearn_FLAG , logger, FORMAT=False):
    """Load and prepare datasets for training and evaluation."""
    logger.info("Loading datasets from MUSE-News...")
    
    # Load dataset with specific subset
    dataset = load_dataset("muse-bench/MUSE-News", name="raw")  # Load raw subset

    # Create dataset splits according to MUSE-News specifications
    if not Unlearn_FLAG:
        raw_datasets = DatasetDict({
            'train': concatenate_datasets([dataset['retain1'], dataset['retain2'], dataset['forget']]), 
            'validation': dataset['holdout'],
            'forget': dataset['forget']  # Optional: For unlearning experiments
        })
    else:
        raw_datasets = DatasetDict({
            'train': concatenate_datasets([dataset['retain1'], dataset['retain2']]), 
            'validation': dataset['holdout'],
            'forget': dataset['forget']  # Optional: For unlearning experiments
        })

    # Special formatting for different subsets
    def format_text(example):
        # Customize formatting based on your task
        return {"text": f"News Article: {example['text']}\n\n"}  # Adjust according to actual dataset structure

    if FORMAT:
        formatted_datasets = raw_datasets.map(format_text)
    else:
        formatted_datasets = raw_datasets

    
    # Log dataset statistics
    logger.info(f"Train dataset size: {len(formatted_datasets['train'])} examples")
    logger.info(f"Evaluation dataset size: {len(formatted_datasets['validation'])} examples")
    
    # Sample and inspect data examples
    logger.info(f"Sample training example:\n{formatted_datasets['train'][0]['text'][:500]}...")
    
    return formatted_datasets


def setup_model_and_tokenizer(model_name, use_8bit, use_4bit, logger):
    """Set up model and tokenizer with optional quantization."""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Add padding token if needed
    if tokenizer.pad_token is None:
        logger.info("Adding [PAD] token to tokenizer")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Configure quantization if requested
    if use_4bit:
        logger.info("Using 4-bit quantization for model loading")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        logger.info("Using 8-bit quantization for model loading")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        logger.info("Loading model in full precision")
        quantization_config = None
    
    # Load model with quantization config if specified
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        # torch_dtype=torch.float16 if not (use_4bit or use_8bit) else None,
    )
    
    return model, tokenizer


# -----------------------------------------------------------------------------
# Preprocessing: Tokenization
# -----------------------------------------------------------------------------
def preprocess_datasets(formatted_datasets, tokenizer, max_length, logger):
    """Tokenize and prepare datasets for training."""
    logger.info(f"Tokenizing datasets with max_length={max_length}")
    
    def tokenize_function(examples):
        tokenizer.truncation_side = "right"  # Ensure truncation is on the right side
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,  # Matches MUSE-News verbmem specification
            padding="max_length",
            add_special_tokens=True,
            return_attention_mask=True,  
            return_tensors="np",    
            padding_side="right"
        )
    
    # Tokenize training dataset
    logger.info("Tokenizing training/evaluating dataset...")
    tokenized_datasets = formatted_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_datasets["train"].column_names,
        desc="Tokenizing training/evaluating dataset",
    )
    
    logger.info(f"Tokenized train dataset size: {len(tokenized_datasets['train'])}")
    logger.info(f"Tokenized eval dataset size: {len(tokenized_datasets['validation'])}")
    
    return tokenized_datasets

def compute_metrics(eval_preds):
    """Compute evaluation metrics (perplexity) from predictions."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate perplexity
    # Only consider positions where labels aren't -100 (padding)
    mask = labels != -100
    loss = np.mean(np.where(mask, (predictions != labels), 0))
    perplexity = math.exp(loss)
    
    return {
        "perplexity": perplexity
    }


def main(args):
    """Main training function."""
    # Set up logging
    logger = setup_logging(args.log_folder, args.slurm_name, args.slurm_id)
    
    logger.info("Starting Llama-2 fine-tuning script")
    logger.info(f"Arguments: {args}")
    
    # Load datasets
    formatted_datasets = load_and_prepare_datasets(args.Unlearn, logger)
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name, args.use_8bit, args.use_4bit, logger
    )
    
    # Preprocess datasets
    tokenized_datasets= preprocess_datasets(
        formatted_datasets, tokenizer, args.max_length, logger
    )
    
    # Set up data collator
    logger.info("Setting up DataCollatorForLanguageModeling")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
    )
    
    # Configure training arguments
    logger.info("Configuring training arguments")

    # Training arguments with original specifications
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Effective batch size = 32
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=20,
        logging_steps=args.logging_steps,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        load_best_model_at_end=True,
        report_to="none",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # Initialize trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[OptimizerCheckpointCallback(save_total_limit=training_args.save_total_limit)
                   ],
    )
    
    # Start training
    logger.info("Starting training process")
    try:
        # Attempt to resume training
        logger.info("Attempting to resume from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        logger.error(f"Failed to resume training: {e}")
        logger.info("Starting fresh training run")
        trainer.train()
    
    # Save final model
    logger.info("Training completed, saving final model")
    trainer.save_model(args.final_model_dir)
    logger.info(f"Model saved to {args.final_model_dir}")
    
    # Save tokenizer alongside model
    tokenizer.save_pretrained(args.final_model_dir)
    logger.info(f"Tokenizer saved to {args.final_model_dir}")
    
    # Run a final evaluation
    logger.info("Running final evaluation")
    final_metrics = trainer.evaluate()
    logger.info(f"Final evaluation metrics: {final_metrics}")
    
    logger.info("Fine-tuning process completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune Llama-2 on MUSE-News')
    
    # Model and output configuration
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", 
                        help="Hugging Face model name or path")
    parser.add_argument('--output_dir', type=str, default="/ibex/project/c2283/Llama-2-7b-ft-muse-news/llama2-7b-muse-news-original", 
                        help="Directory to save checkpoints and logs")
    parser.add_argument('--final_model_dir', type=str, default="./llama2-7b-muse-bench-news-original-final", 
                        help="Directory to save the final model")
    
    # Training configuration
    parser.add_argument('--eval_steps', type=int, default=50,
                       help="Evaluation frequency in steps")
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help="Learning rate for optimizer")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size per device for training")
    parser.add_argument('--num_epochs', type=float, default=5, 
                        help="Number of training epochs")
    parser.add_argument('--max_length', type=int, default=2048, 
                        help="Maximum sequence length for tokenization")
    parser.add_argument('--logging_steps', type=int, default=50, 
                        help="Log training metrics every N steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help="Number of update steps to accumulate gradients for")
    parser.add_argument('--Unlearn', action='store_true', 
                        help="Enable update training sets for exact unlearning")

    
    # Quantization options
    parser.add_argument('--use_8bit', action='store_true', 
                        help="Load model in 8-bit quantization")
    parser.add_argument('--use_4bit', action='store_true', 
                        help="Load model in 4-bit quantization (overrides 8-bit if both specified)")
    
    # Logging configuration
    parser.add_argument('--log_folder', type=str, default="././llama2-7b-muse-bench-news-original", 
                        help="Folder to store logs")
    parser.add_argument('--slurm_name', type=str, default='slurm', 
                        help="Slurm job name for log files")
    parser.add_argument('--slurm_id', type=str, default='00000000', 
                        help="Slurm job ID for log files")
    
    args = parser.parse_args()

    if args.Unlearn:
        # repalce path in args.output_dir and args.final_model_dir
        args.output_dir = args.output_dir.replace("original", "unlearn")
        args.final_model_dir = args.final_model_dir.replace("original", "unlearn")

    args.log_folder = args.output_dir

    main(args)