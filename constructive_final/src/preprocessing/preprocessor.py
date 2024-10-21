import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from wordsegment import load, segment
from nltk.corpus import words
import multiprocessing as mp
from multiprocessing import cpu_count
import re, time

# Download required NLTK resources
nltk.download('punkt')
nltk.download('words')

load()

class Preprocessor:
    
    def __init__(self):
        self.english_words = set(words.words())  # Set of valid English words
        pass
        
    def preprocess(self, text):
        """Preprocess the text by tokenizing and segmenting."""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        tokens = word_tokenize(text)
        preprocessed = []
        
        for token in tokens: 
            # If token is not in English words, try segmenting it
            if token not in self.english_words: 
                segmented_tokens = segment(token)
                preprocessed.extend(segmented_tokens)
            else:
                preprocessed.append(token)
                
        return " ".join(preprocessed)

def run_normalize(df):
    """Apply preprocessing to the DataFrame."""
    pp = Preprocessor()    
    df['pp_comment_text'] = df['comment_text'].apply(pp.preprocess)
    return df

def parallelize(df, func):
    """Parallelize the preprocessing using multiprocessing."""
    cores = cpu_count()
    data_split = np.array_split(df, cores)
    pool = mp.Pool(cores)
    df_processed = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return df_processed

if __name__ == "__main__":
    # Load your dataset
    C3_feats_df = pd.read_csv("./data/processed/C3_anonymized_origin.csv")
    C3_feats_df = C3_feats_df.head(10)
    start = time.time()
    print('Start time: ', start)
    
    # Preprocess the rows in parallel 
    df_processed = parallelize(C3_feats_df, run_normalize)
    
    end = time.time()
    print('Total time taken: ', end-start)    
    
    # Save the processed DataFrame
    #df_processed.to_csv("./data/processed/C3_anonymized.csv", index=False)
    print('The preprocessed features file written: ', "./data/processed/C3_anonymized.csv")
