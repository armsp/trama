# trama
Trope detection using LLaMA

## Tasks
- [x] Filter Tropes to study
- [ ] Check how many of these tropes are inherently understood by the LLM
- [ ] Make the dataset of stories where there are tropes that you have filtered, most importantly the tropes that the LLM does not understand.
- [ ] Test for the trope without finetuning - just ask
- [ ] Prompt with description and then ask
- [ ] Finetune and then ask

How to decide between Prompting and Finetuning?

## Stories Dataset
1. New Yorker Flash Fiction stories - around 35+
2. Reedsy Flash Fiction - https://reedsy.com/discovery/blog/best-flash-fiction
3. https://tinhouse.com/category/fiction/ - not yet scraped
4. https://www.flashfictiononline.com/ - not yet scraped
5. 


U+2019 character needs to be replaced with regular apostrophe - '

## Running LLaMA 2 7b
Make the language output maximum sequence it can
LLaMA prompt tuning?
How to make it output everything it knows in multiple iterations since probably one output is not enough??
Check how many tropes it knows
Check how many it can identify prior to any thing by inputting it stories -> Look below for how

"Turn On" the model
iterate over the stories and ask it to list all the tropes it can identify and to give reasons for each identified trope
---
All models support sequence length up to 4096 tokens, but we pre-allocate the cache according to max_seq_len and max_batch_size values. So set those according to your hardware.
These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

## Prompting vs Finetuning


## Immediate TODO
- LLaMA base test for a certain thing.
- QLORA fine tuning
- Test again

LLaMA Models
- [ ] 7B
- [ ] 13B
- [ ] 70B

## Time Distribution
Dataset Preparation and gathering: ~9 hours
LLaMA Research: 
LLaMA Prompting vs Fine tuning experiment: 
Testing and Validation:

# Folders
llama_tokenizer contains tokenizer model files cloned from huggingface without the actual models

# Euler 
Contains the scripts used to run the LLaMA 2 trope extraction scripts, including all the details about environment setup and job submission script

# universal-sentence-encoder-large_5
contains the use model from Google

# julyter notebooks-
1. `LLaMA.ipynb` - 
2. `select tropes.ipynb` -
3. `semantic_search.ipynb` - 
4. `story_dataset_maker.ipynb` - 
5. `story_summaries.ipynb` - 
6. `story_vectors.ipynb` - 
7. `trope_examples_dataset.ipynb` -
8. `trope_vectors.ipynb` - 
9. `visuals.ipynb` - 

