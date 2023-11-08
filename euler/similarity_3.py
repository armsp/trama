import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import scipy.spatial
import tensorflow_hub as hub
import tensorflow as tf
# from numpy import dot
# from numpy.linalg import norm
print("loading model...")
model_path = "/cluster/work/lawecon/Work/raj/use/large_5"
encoder = hub.load(model_path)
print("model loaded")
stories = pd.read_csv('stories_with_summary.csv')
print(stories)
trope_examples = pd.read_csv('trope_examples.csv')#, dtype={"embedding": })
trope_examples = trope_examples.dropna()
print("read csvs")

matrix = np.load('trope_examples_embeddings.npy')
print(matrix.shape)
print("embeddings matrix ready")
task = "summary_gpt_3_5"
task = "story"
task = "summary_gpt_3_5_t_14"
# loop through stories/summaries
for story in tqdm(stories.itertuples()):
    print(f"processing story - {story.sid}")
    # make story/summary matrix
    story_embed = encoder([story.summary_gpt_3_5_t_14])
    story_embed = np.tile(story_embed, (matrix.shape[0],1))
    print(story_embed.shape)
    # similarity = 1 - scipy.spatial.distance.cdist(matrix, story_embed, 'cosine')
    cosine_similarities = tf.reduce_sum(tf.multiply(matrix, story_embed), axis=1)
    trope_examples[story.sid] = cosine_similarities

print("saving csv")
# drop the trope_embeddings column and then save the df
# trope_examples = trope_examples.drop('embed', axis=1)
trope_examples.to_csv(f'/cluster/home/sharaj/llama/similarity_{task}.csv', index=False)