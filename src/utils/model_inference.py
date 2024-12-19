# load modules
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


def setup_generator_pipe():
    # set up generator pipeline
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

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
        max_new_tokens=128,
    )

    return generator


def run_inference(prompt, generator):
    

    # apply chat template and add generation prompt
    #prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)

    # get response from model
    response = generator(prompt)[0]["generated_text"][-1]
    generated_text = response.get("content")

    return generated_text