import pandas as pd
import torch
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
# Load the LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token or add a custom one
    # Alternatively, add a new padding token:
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))

# Load your CSV file
csv_file_path = '/home/product_master/Bhumika/Artiflex/christmas_banner_unique_prompts.csv'
csv_data = pd.read_csv(csv_file_path)

# Split data into training and testing sets
train_data, test_data = train_test_split(csv_data, test_size=0.2, random_state=42)

# Custom Dataset class
class PromptHTMLDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.csv_data = csv_file
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        prompt = str(self.csv_data.iloc[idx]['prompt'])
        html_output = str(self.csv_data.iloc[idx]['html_output'])

        # Tokenize the prompt (input)
        inputs = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        # Tokenize the html_output (target labels)
        labels = self.tokenizer(html_output, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").input_ids

        # Replace padding token id's with -100 so they are ignored during loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(), 
            'attention_mask': inputs['attention_mask'].squeeze(), 
            'labels': labels.squeeze()
        }

# Create dataset objects for train and test sets
train_dataset = PromptHTMLDataset(train_data, tokenizer)
test_dataset = PromptHTMLDataset(test_data, tokenizer)

# Define LoRA configuration
lora_config = LoraConfig(
    r=4,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # Bias handling
    target_modules=["q_proj", "v_proj"]  # Specify the modules to apply LoRA on
)

# Load the model in 8-bit precision with device mapping and apply LoRA
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)
model.config.use_memory_efficient_attention = True
import transformers
class CustomCallback(transformers.TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Access training loss from the log history
        if state.log_history and 'loss' in state.log_history[-1]:
            train_loss = state.log_history[-1]['loss']
            print(f"Epoch {state.epoch}: Train Loss: {train_loss:.4f}")
        else:
            print(f"Epoch {state.epoch}: Train Loss: Not available")


# Define your training arguments
training_args = TrainingArguments(
    output_dir='./results',         # output directory
    evaluation_strategy='epoch',    # evaluate after each epoch
    save_strategy='epoch',          # save after each epoch
    logging_dir='./logs',           # directory for storing logs
    logging_steps=10,               # log every 10 steps
    load_best_model_at_end=True,    # load the best model at the end
    metric_for_best_model='eval_loss',  # use eval_loss to determine best model
    per_device_train_batch_size=2,  # adjust based on your GPU capacity
    num_train_epochs=100,             # number of training epochs
    fp16=True,  # Enable mixed precision
    gradient_accumulation_steps=4,
)

# Define a simple metric function to compute evaluation loss
def compute_metrics(eval_pred):
    loss = eval_pred.loss
    return {'eval_loss': loss}

# Initialize the Trainer with your model, training arguments, and custom callback
trainer = Trainer(
    model=model,                      # your model
    args=training_args,
    train_dataset=train_dataset,      # your training dataset
    eval_dataset=test_dataset,        # your evaluation dataset
    compute_metrics=compute_metrics,  # your metrics function
    callbacks=[CustomCallback()],     # add custom callback here
)
torch.cuda.empty_cache()
# Start training
trainer.train()
