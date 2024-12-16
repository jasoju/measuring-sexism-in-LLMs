# load modules
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

def run_inference(prompt):

    # set up generator pipeline
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    # check GPU
    print(torch.cuda.is_available())

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

    def get_response(prompt):
        response = generator(prompt)
        text = response[0]["generated_text"]
        return text

    prompt = "Explain the concept of GenAI."
    model_response = get_response(prompt)
    
    return model_response