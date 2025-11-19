import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "Qwen/Qwen3-4B-Instruct-2507"
cache_dir = "C:\\Users\\ndela\\.cache\\huggingface\\hub"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map="auto",  # Automatically uses GPU
)


# Chat function
def chat(messages, max_new_tokens=512, temperature=0.7):
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response


# Example usage
messages = [
    {"role": "user", "content": "You are a pirate. What is the meaning of life?"},
]

response = chat(messages)
print(response)
