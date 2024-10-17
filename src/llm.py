import pandas as pd
import torch
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import logging  # Import the logging module
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
import os
import logging
import gc
from transformers import AutoConfig
import transformers


import torch
# Create a dummy tensor
dummy_tensor = torch.tensor([1.0]).cuda()

# Check if NCCL is available
is_nccl_available = torch.cuda.nccl.is_available(dummy_tensor)
print(f"NCCL available: {is_nccl_available}")
# Initialize distributed process group
dist.init_process_group(backend='nccl')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training.log')

# Load the LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set device based on local rank (for multi-GPU training)
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Load your CSV file
csv_file_path = '/home/drovco/Bhumika/Artiflex/christmas_banner_unique_prompts.csv'
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

        # Replace padding token ids with -100 so they are ignored during loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(), 
            'attention_mask': inputs['attention_mask'].squeeze(), 
            'labels': labels.squeeze()
        }

# Create dataset objects for train and test sets
train_dataset = PromptHTMLDataset(train_data, tokenizer)
test_dataset = PromptHTMLDataset(test_data, tokenizer)

# Create distributed samplers for multi-GPU training
train_sampler = DistributedSampler(train_dataset)
test_sampler = DistributedSampler(test_dataset)

# Create DataLoaders using distributed samplers
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=2, num_workers=4)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=2, num_workers=4)

# Define LoRA configuration
lora_config = LoraConfig(
    r=4,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # Bias handling
    target_modules=["q_proj", "v_proj"]  # Specify the modules to apply LoRA on
)

# Load the model in 8-bit precision with device mapping and apply LoRA
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)

# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)
model.config.use_memory_efficient_attention = True

# Custom Callback to log training loss
class CustomCallback(transformers.TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history and 'loss' in state.log_history[-1]:
            train_loss = state.log_history[-1]['loss']
            logging.info(f"Epoch {state.epoch}: Train Loss: {train_loss:.4f}")
        else:
            logging.info(f"Epoch {state.epoch}: Train Loss: Not available")
        torch.cuda.empty_cache()
        gc.collect()

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    evaluation_strategy="no",  # Keep evaluation strategy as "no"
    logging_steps=1,
    metric_for_best_model='eval_loss',  # You can keep this if you want to monitor this metric, but it won't have an effect without evaluation
    per_device_train_batch_size=1,  # Adjust per-device batch size for distributed training
    num_train_epochs=100,
    fp16=True,
    gradient_accumulation_steps=2,
    dataloader_num_workers=2,
)
# Define a simple metric function to compute evaluation loss
def compute_metrics(eval_pred):
    loss = eval_pred.loss
    return {'eval_loss': loss}

class SaveModelCallback(transformers.TrainerCallback):
    def __init__(self, model_name):
        self.model_name = model_name  # Store model_name for accessing in the callback

    def on_epoch_end(self, args, state, control, **kwargs):
        tokenizer = kwargs.get('tokenizer', None)
        if tokenizer is None:
            logging.error("Tokenizer is not available in the callback.")
            return

        if state.epoch % 10 == 0 and state.epoch > 0:
            output_dir = f"./results/checkpoint-epoch-{int(state.epoch)}"
            kwargs['model'].save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Save model configuration manually
            model_config = AutoConfig.from_pretrained(self.model_name)
            model_config.save_pretrained(output_dir)

            logging.info(f"Model checkpoint saved at {output_dir}")
            # Check if config.json exists after saving
            if os.path.exists(os.path.join(output_dir, 'config.json')):
                logging.info(f"Config file saved successfully at {output_dir}/config.json")
            else:
                logging.warning(f"Config file NOT found after saving at {output_dir}")

# Function to move model to CPU for evaluation
def move_model_to_cpu(trainer):
    trainer.model.to('cpu')

# Function to move model back to GPU for training
def move_model_to_gpu(trainer, device):
    trainer.model.to(device)

# Initialize the Trainer with your model, training arguments, and custom callback
trainer = Trainer(
    model=model,                      
    args=training_args,
    train_dataset=train_dataset,      
    eval_dataset=test_dataset,        
    compute_metrics=compute_metrics, 
    tokenizer=tokenizer , 
    callbacks=[CustomCallback(), SaveModelCallback(model_name="meta-llama/Llama-3.2-3B")],    
)
trainer.tokenizer = tokenizer
# Start training on the GPU
trainer.model.to(local_rank)
trainer.train()

# Move model to CPU for evaluation
move_model_to_cpu(trainer)
eval_results = trainer.evaluate()
torch.cuda.empty_cache()

# Print evaluation results
print(f"Evaluation Results on CPU: {eval_results}")

# Move model back to GPU after evaluation (optional for further training)
move_model_to_gpu(trainer, local_rank)
