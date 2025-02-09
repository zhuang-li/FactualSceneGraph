import argparse
import wandb
from datasets import load_dataset
from transformers import TrainerCallback, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch

class LogLearningRateCallback(TrainerCallback):
    """
    Callback that logs the learning rate at the end of each step.
    """
    def on_step_end(self, args, state, control, **kwargs):
        optimizer = kwargs.get('optimizer')
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr}, step=state.global_step)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a scene graph generation model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lizhuang144/FACTUAL_Scene_Graph",
        help="Dataset identifier from Hugging Face hub",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="google/flan-t5-base",
        help="Model checkpoint to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models/factual_sg/",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=200, help="Number of steps between evaluations")
    parser.add_argument("--generation_max_length", type=int, default=512, help="Maximum length of generated sequences")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    return parser.parse_args()

def prepare_examples(example):
    """
    Prepares each example by creating an 'input_text' with a task prompt and setting the 'target_text'.
    """
    example["input_text"] = "Generate Scene Graph: " + example["caption"]
    example["target_text"] = example["scene_graph"]
    return example

def preprocess_function(examples, tokenizer):
    """
    Tokenizes the input and target texts without fixed padding.
    Dynamic padding will be applied at the batch level by the data collator.
    """
    inputs = examples["input_text"]
    targets = examples["target_text"]

    # Tokenize without fixed padding
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding=False
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=512, truncation=True, padding=False
        )
    # Replace all pad tokens in the labels with -100 so they are ignored in the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]

    # Optional: print lengths for debugging (comment out in production)
    print(len(model_inputs["input_ids"][0]), len(labels["input_ids"][0]))
    return model_inputs

def main():
    args = parse_args()

    # Initialize Weights & Biases
    project_name = "FACTUAL_Scene_Graph"
    run_name = (
        f"dataset={args.dataset}-checkpoint={args.checkpoint}-num_epochs={args.num_epochs}-"
        f"batch_size={args.batch_size}-learning_rate={args.learning_rate}-seed={args.seed}-eval_steps={args.eval_steps}"
    )
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "eval_steps": args.eval_steps,
        },
    )

    # Use CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.checkpoint, trust_remote_code=True
    ).to(device)

    # Load and prepare the dataset
    dataset = load_dataset(args.dataset)
    dataset = dataset.map(prepare_examples)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
    # Create a 90/10 train-test split from the available 'train' split
    tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=args.seed)

    # Use a data collator that pads to the longest sequence in the batch
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_steps,
        generation_max_length=args.generation_max_length,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        seed=args.seed,
        overwrite_output_dir=True,
        save_total_limit=1,
        report_to="wandb",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogLearningRateCallback()],
    )

    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    main()
