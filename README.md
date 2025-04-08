# Measuring ambivalent sexism in large language models: A validation study

In this master thesis project, I aim to systematically validate the Ambivalent Sexism Inventory for LLMs by evaluating reliablilty - through internal consistency, alternate-form reliability, and option-order symmtery - and validity - through concurrent validity, convergent validity, and factorial validity.

[collect_data.py](src/collect_data.py) is used to collect model responses to different tasks. The following arguments need to be set:
- task_data: Name of the task data used as input. Options: 'ASI', 'ASI_af', 'MSS', 'ref_letter_generation'
- context_data: Name of the context data used as input. Options: None, 'chatbot_arena_conv', 'persona_hub'. 'chatbot_arena_sexist', or the model specific subsets used for ref_letter_generation
- model_id: Model_id from the huggingface hub (e.g. meta-llama/Llama-3.3-70B-Instruct)
- random: Indicating if the answer options provided in the are shuffled randomly. Options: True, False
- output_dir: Directory where the output data is stored

All collected data can be found [here](src/output_data/).

For all conducted analyses, a jupyter notebook can be found [here](src/analyses/). This includes:
- [the descriptive analyses](src/analyses/descriptives.ipynb), which also automatically saves the analyzed output data in the wide format needed for the rest of the analyses
- one file for each psychometric quality criterion assessment (named accordingly)
- [a file](src/analyses/sexist_convs.ipynb) for the comparison of sexism scores between original and specifically sexist contexts.

  This repo also contains the latex source files and pdf file of my thesis [here](src/document).
