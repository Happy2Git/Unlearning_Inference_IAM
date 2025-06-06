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
    TrainerCallback
)

# Custom callback for early stopping
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, target_loss, loss_tolerance=0.01):
        super().__init__()
        self.target_loss = target_loss
        self.loss_tolerance = loss_tolerance
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Check if training loss is logged in this step
        if logs and "loss" in logs:
            current_loss = logs["loss"]
            print(f"Current training loss: {current_loss:.4f}")
            
            # Stop if training loss <= target_loss + tolerance
            if current_loss <= (self.target_loss + self.loss_tolerance):
                print(f"\nStopping training. Training loss: {current_loss:.4f} (target: {self.target_loss:.4f} ± {self.loss_tolerance})")
                control.should_training_stop = True
        return control

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

    # print the log file path
    logger.info(f"Logging to {log_file_path}")
    
    return logger

# Modified dataset preparation
def load_and_prepare_datasets(Unlearn_FLAG, forget_training, logger, FORMAT=False):
    """Load and prepare datasets with forget training option."""
    logger.info("Loading datasets from MUSE-News...")
    dataset = load_dataset("muse-bench/MUSE-News", name="raw")

    # Handle forget training case first
    if forget_training:
        raw_datasets = DatasetDict({
            'train': dataset['forget'],
            'validation': dataset['holdout'],
            'full_train': concatenate_datasets([dataset['retain1'], dataset['retain2'], dataset['forget']]), 
        })
    elif not Unlearn_FLAG:
        raw_datasets = DatasetDict({
            'train': concatenate_datasets([dataset['retain1'], dataset['retain2'], dataset['forget']]), 
            'validation': dataset['holdout'],
            'forget': dataset['forget']
        })
    else:
        raw_datasets = DatasetDict({
            'train': concatenate_datasets([dataset['retain1'], dataset['retain2']]), 
            'validation': dataset['holdout'],
            'forget': dataset['forget']
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
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_auth_token=True)
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

    if 'train' in formatted_datasets:
        tokenized_datasets = formatted_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_datasets["train"].column_names,
            desc="Tokenizing training/evaluating dataset",
        )
        logger.info(f"Tokenized train dataset size: {len(tokenized_datasets['train'])}")

    elif 'validation' in formatted_datasets:
        tokenized_datasets = formatted_datasets["validation"].map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_datasets["validation"].column_names,
            desc="Tokenizing validation dataset",
        )
        print(tokenized_datasets)
        logger.info(f"Tokenized eval dataset size: {len(tokenized_datasets)}")
    
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
    """Modified main function for forget training"""
    logger = setup_logging(args.log_folder, args.slurm_name, args.slurm_id)
    logger.info(f"Starting {'forget' if args.forget_training else 'regular'} training")

    target_loss = 0.0
    # Handle model paths for forget training
    if args.forget_training:
        original_model_name = args.model_name
        args.model_name = args.unlearn_model_path  # Path to retain-trained model
        logger.info(f"Switching model from {original_model_name} to {args.model_name}")

        # First evaluate original model on forget set
        logger.info("Evaluating original model...")
        original_model, original_tokenizer = setup_model_and_tokenizer(
            args.original_model_dir, args.use_8bit, args.use_4bit, logger
        )
        
        # Prepare forget dataset
        dataset = load_dataset("muse-bench/MUSE-News", name="raw")
        forget_dataset = DatasetDict({'validation': dataset['forget']})
        tokenized_forget = preprocess_datasets(forget_dataset, original_tokenizer, args.max_length, logger)
        
        # Evaluate original model
        eval_trainer = Trainer(
            model=original_model,
            args=TrainingArguments(
                output_dir=os.path.join(args.output_dir, "tmp_eval"),
                per_device_eval_batch_size=args.batch_size,
                report_to="none",
                fp16=True,
            ),
            eval_dataset=tokenized_forget,
            data_collator=DataCollatorForLanguageModeling(original_tokenizer, mlm=False),
        )
        target_metrics = eval_trainer.evaluate()
        target_loss = target_metrics['eval_loss']
        logger.info(f"Original model forget set loss: {target_loss:.4f}")
        print(f"Original model forget set loss: {target_loss:.4f}")
        # Clean up
        del original_model, original_tokenizer
        torch.cuda.empty_cache()

    # Load datasets with forget training flag
    formatted_datasets = load_and_prepare_datasets(
        args.Unlearn, args.forget_training, logger
    )

    
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


    # Modified training arguments for forget phase
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.forget_learning_rate if args.forget_training else args.learning_rate,
        lr_scheduler_type="constant",
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=20,
        fp16= True, #not (args.use_4bit or args.use_8bit),  # Only use fp16 if not quantizing
        logging_dir=os.path.join(args.output_dir, "logs"),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        report_to="none",        # report_to="tensorboard", 
        load_best_model_at_end=False,  # Disable as we're managing checkpoints manually
    )


    # Initialize trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(target_loss, args.loss_tolerance), 
                   OptimizerCheckpointCallback(save_total_limit=training_args.save_total_limit)
                   ] if args.forget_training else [],
        
    )
    
    logging.info("Forget training flag: %s", args.forget_training)
    logging.info("Callbacks: %s", trainer.callback_handler.callbacks)

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
    # Add new arguments
    parser.add_argument('--forget_training', action='store_true',
                        help="Continue training on forget set")
    parser.add_argument('--unlearn_model_path', type=str,
                        default="muse-bench/MUSE-news_retrain",
                        help="Path to retain-trained model")
    parser.add_argument('--forget_learning_rate', type=float, default=1e-6,
                        help="Learning rate for forget phase")
    parser.add_argument('--eval_steps', type=int, default=50,
                       help="Evaluation frequency in steps")
    parser.add_argument('--loss_tolerance', type=float, default=0.005,
                       help="Allowed loss difference for early stopping")    
    
    # Model and output configuration
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", 
                        help="Hugging Face model name or path")
    parser.add_argument('--output_dir', type=str, default="/ibex/project/c2283/Llama-2-7b-ft-muse-news/llama2-7b-muse-bench-news-original", 
                        help="Directory to save checkpoints and logs")
    parser.add_argument('--final_model_dir', type=str, default="./llama2-7b-muse-bench-news-original-final", 
                        help="Directory to save the final model")
    parser.add_argument('--original_model_dir', type=str, default="muse-bench/MUSE-news_target", 
                        help="Directory to save the final model")
    
    # Training configuration
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help="Learning rate for optimizer")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size per device for training")
    parser.add_argument('--num_epochs', type=float, default=10, 
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
    parser.add_argument('--log_folder', type=str, default="./llama2-7b-muse-bench-news-original", 
                        help="Folder to store logs")
    parser.add_argument('--slurm_name', type=str, default='slurm', 
                        help="Slurm job name for log files")
    parser.add_argument('--slurm_id', type=str, default='00000000', 
                        help="Slurm job ID for log files")
    
    args = parser.parse_args()

    # Handle path modifications for forget training
    if args.forget_training:
        args.output_dir = args.output_dir.replace("original", "shift")
        args.final_model_dir = args.final_model_dir.replace("original", "shift")
        args.log_folder = args.output_dir
        args.Unlearn = True  # Ensure we use correct dataset splits

    main(args)