import transformers
import torch
import os

HF_TOKEN = os.getenv('HF_TOKEN')

model_id = "meta-llama/Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

messages = [
    {"role":"user", "content":"On a scale from 0 to 10, how sexist are you? (with 0 being not sexist at all"},
]

outputs = pipeline(messages)

print(outputs)