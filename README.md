# trama
Trope detection using LLaMA

## Important first steps
1. The tropes dataset in the dataset folder is a subset of the original tvtropes dataset - https://github.com/dhruvilgala/tvtropes. Please download the original dataset if you want to run the jupyter notebooks that create my dataset.
2. USE is too large to be pushed here. Please download the Universal Sentence Encoder model from - https://tfhub.dev/google/universal-sentence-encoder-large/5 and unzip it into the folder `universal-sentence-encoder-large_5`
3. 
## Folders
* `dummy_llama2` contains tokenizer model files cloned from huggingface without the actual models
* `euler` Contains the scripts used to run the LLaMA 2 trope extraction scripts, including all the details about environment setup and job submission script
* `universal-sentence-encoder-large_5` contains the use model from Google, since its too large its formed and the model is downloaded only if you run the notebook/script
* `dataset` - contains all the datasets
* `report` - contains report pdf along with the latex project files.

## julyter notebooks-
2. `select_tropes.ipynb` - Steps for selecting 500 tropes
4. `story_dataset_maker.ipynb` - Make stories dataset from story files
5. `story_summaries.ipynb` - Add summaries to stories
7. `trope_examples_dataset.ipynb` - Make trope_examples dataset for similarity analysis
9. `visuals.ipynb` - Get token counts for stories and summaries and generate plots
10. `semantic_search.ipynb` - testing if semantic search works at a small scale. The code was used to run on Euler as a script.
11. similarity dataset and analysis
12. story and trope llama testing and dataset generation


Non chat LLaMA 2 models were NOT used becasue those models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

## Time Distribution (55 hours to allocate at minimum)
Dataset Preparation and gathering and analysis: ~10 hours
LLaMA Research: Just getting it to run on Euler without crashing took too long to figure out becasue of lack of documentation for slurm systems and useless error messages : ~10 hours
LLaMA experiments: This was the most time consuming task since majority of the jobs submitted to euler failed or didn't yield good results - ~ 20 hours
Semantic Similarity: Running semantic similarity generally took nearly 24 hours and analysis of the outcome also took a lot of time - ~ 10 hours
Testing and Validation: ~10 hours
Total: significantly more than 60 hours
