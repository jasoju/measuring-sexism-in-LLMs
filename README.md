# Can a psychological assessment measure sexism in LLMs? A validation study of the Ambivalent Sexism Inventory

In this master thesis project I aim to systematically validate the Ambivaent Sexism Inventory for LLMs. The thesis can be found [here](thesis.pdf). I also presented this work at IC2S2 2025. The poster can be found below.

## Abstract
Large language models (LLMs) often reflect gender biases from their training data, making it crucial to develop reliable and valid methods for measuring these biases. Existing approaches have been criticized for inconsistencies in how gender bias is conceptualized and operationalized. This thesis investigates whether the Ambivalent Sexism Inventory (ASI), a well-established psychological test, can be used to measure sexism in LLMs. We administer the ASI to six state-of-the-art LLMs and conduct a systematic validation by evaluating reliability – through internal consistency, alternate-form reliability, and option-order symmetry – and validity – through concurrent validity, convergent validity, and factorial validity. To approximate psychometric testing conditions, we conceptualize an LLM as a representation of a population and induce individuals by prompting the model with different context information. Two context types are employed: human-chatbot interactions and personas. In all cases, we find low reliability or low validity of the ASI. These findings show that the ASI is not a valid measure for any of the six LLMs tested. This also entails no significant positive correlation between the ASI score and the use of sexist language in a downstream task. This underscores the importance of conducting validation studies before interpreting psychological test scores in the context of LLMs. However, our results also show that the method used to induce individuals influences the evaluation outcomes of psychometric quality criteria. This raises fundamental questions about the generalizability of results across context types and how human-centered psychological concepts, such as “individuals”, should be conceptualized in the LLM domain.

![Image](https://github.com/user-attachments/assets/7570d8c0-bb26-418a-a583-aafed404fe6d)


## Code
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

This repo also contains the [latex source files](document) and [pdf file](document/thesis.pdf) of my thesis.
