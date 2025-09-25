
def run_inference(llm, tokenizer, sampling_params, prompts, model_id) -> list[str]:
    # for all models except Centaur apply chat template (prompts are already in list format)
    if "Centaur" not in model_id:   
        prompts = tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
        )

    # print for debugging
    print(f"---------- seed: {sampling_params.seed} ----------")
    print("max_tokens:", sampling_params.max_tokens)

    # generate model responses
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    responses = [output.outputs[0].text for output in outputs]

    return responses