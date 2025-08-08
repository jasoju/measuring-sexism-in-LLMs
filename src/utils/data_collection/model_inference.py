# load modules
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, set_seed
from datasets import Dataset
import torch
import pandas as pd
from tqdm import tqdm


def setup_generator_pipe(model_id:str) -> transformers.TextGenerationPipeline:
    # set max_new_tokens based on task
    max_new_tokens = 20 
    
    # set up generator pipeline
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens
    )

    return generator


def run_inference(row:pd.Series, generator:transformers.TextGenerationPipeline) -> str:
    # get response from model
    if "random_state" in row:
        set_seed(row["random_state"])
        response = generator(row["prompt"], do_sample=True)[0]["generated_text"][-1].get("content")
    else:
        response = generator(row["prompt"], do_sample=False)[0]["generated_text"][-1].get("content")

    return response