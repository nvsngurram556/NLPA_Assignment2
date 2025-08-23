from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
# Load the dataset
dataset = load_dataset("json", data_files="data/processed/QandA.json")

import torch

# Load the tokenizer and model
model_name = "gpt2"  # Replace if you want a different model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()
# GPT2 doesn't have a pad token by default, so we set it
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Select device: MPS (Apple Silicon), CUDA (NVIDIA), or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = model.to(device)



# Preprocess the dataset
# Preprocessing function
def preprocess_function(examples):
    questions = [str(q) for q in examples["question"]]
    # SQuAD-style: answers is a list of dicts with "text" key (list of strings)
    answers = [str(a["text"][0]) if isinstance(a, dict) and "text" in a and len(a["text"]) > 0 else "" for a in examples["answer"]]

    inputs = [q + tokenizer.eos_token for q in questions]
    targets = [a + tokenizer.eos_token for a in answers]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=512,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the dataset, remove original text columns to avoid collator issues
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["question", "answer"]
)

# Use DataCollatorForSeq2Seq to handle dynamic padding and label -100 automatically
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Setup training arguments
training_args = TrainingArguments(
    output_dir="./sft_model",
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    remove_unused_columns=False,
    report_to="none",
    bf16=True if torch.cuda.is_available() else False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./sft_model")
tokenizer.save_pretrained("./sft_model")
print("Model fine-tuned and saved to ./sft_model")