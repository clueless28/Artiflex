import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = "/home/drovco/Bhumika/Artiflex/results/checkpoint-epoch-10"  # Update this path to the saved checkpoint
#tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set the model to evaluation mode and load it onto the GPU if available
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to generate text predictions
def generate_html(prompt, max_length=128):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    # Generate output from the model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,  # Beam search for better results
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    # Decode the generated tokens
    generated_html = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_html

# Test with a few example prompts
test_prompts = [
    "Generate a banner for a Christmas sale on chocolates.",
    "Create a web layout with a 50% discount offer on electronics.",
    "Design a promotional banner for Black Friday deals on fashion accessories."
]

# Run the model on each test prompt and print the output
for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    generated_html = generate_html(prompt)
    print(f"Generated HTML: {generated_html}\n")
