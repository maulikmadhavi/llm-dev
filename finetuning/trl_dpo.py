import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load data and model
dataset = load_dataset("shawhin/youtube-titles-dpo")

# Check dataset structure
print("Dataset keys:", dataset.keys())
print("Train dataset columns:", dataset['train'].column_names)
print("Sample from train dataset:", dataset['train'][0])
print("---")

# Convert conversation format to string format to avoid chat template issues
def convert_conversations_to_strings(examples):
    prompts = []
    chosen = []
    rejected = []
    
    for prompt, chosen_resp, rejected_resp in zip(examples['prompt'], examples['chosen'], examples['rejected']):
        # Extract content from conversation format
        prompt_text = prompt[0]['content'] if prompt and len(prompt) > 0 else ""
        chosen_text = chosen_resp[0]['content'] if chosen_resp and len(chosen_resp) > 0 else ""
        rejected_text = rejected_resp[0]['content'] if rejected_resp and len(rejected_resp) > 0 else ""
        
        prompts.append(prompt_text)
        chosen.append(chosen_text)
        rejected.append(rejected_text)
    
    return {
        'prompt': prompts,
        'chosen': chosen,
        'rejected': rejected
    }

# Apply conversion to both train and validation sets
dataset['train'] = dataset['train'].map(convert_conversations_to_strings, batched=True)
dataset['valid'] = dataset['valid'].map(convert_conversations_to_strings, batched=True)

print("After conversion - Sample from train dataset:", dataset['train'][0])
print("---")

# Load the model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # set pad token to eos token

# Step2: GEnerate title with BASE model

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",)

def format_chat_prompt(user_input, system_message="You are a helpful assistant."):
    """
    Formats user input into the chat template format with <|im_start|> and <|im_end|> tags.

    Args:
        user_input (str): The input text from the user.

    Returns:
        str: Formatted prompt for the model.
    """
    
    # Format user message
    user_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n"
    
    # Start assistant's turn
    assistant_prompt = "<|im_start|>assistant\n"
    
    # Combine prompts
    formatted_prompt = user_prompt + assistant_prompt
    
    return formatted_prompt

prompt = format_chat_prompt(dataset['valid']['prompt'][0][0]['content'])

outputs = generator(prompt, max_length=100, truncation=True, num_return_sequences=1, temperature=0.1)

print("Generated Title:", outputs[0]['generated_text'])
ft_model_name = model_name.split('/')[1].replace("Instruct", "DPO")

training_args = DPOConfig(
    output_dir=ft_model_name, 
    logging_steps=25,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=1,
)
trainer = DPOTrainer(
    model=model, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
)
trainer.train()
# Save the fine-tuned model
trainer.save_model(ft_model_name)

# Load the fine-tuned model
ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, device_map="auto")

# Set up text generation pipeline
generator = pipeline("text-generation", model=ft_model, tokenizer=tokenizer, device_map="auto")

# Example prompt
prompt = format_chat_prompt(dataset['valid']['prompt'][0][0]['content'])

# Generate output
outputs = generator(prompt, max_length=100, truncation=True, num_return_sequences=1, temperature=0.7)

print(outputs[0]['generated_text'])

print(format_chat_prompt(dataset['valid']['prompt'][0][0]['content']))
