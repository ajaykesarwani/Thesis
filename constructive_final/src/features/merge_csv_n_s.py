import pandas as pd

# Read DataFrames
df = pd.read_csv("E:/constructive code/constructiveness/data/processed/C3_anonymized_fnd_n_s.csv")

# List of columns to keep
columns_to_keep = ['constructive',
                        'Quotes', 'Punc_Freq', 'Punc_Type', 'Exclamation', 'Stopwords', 'Camelcase',
                        'Negation', 'Propn', 'Usermention', 'Hashtags', 'Misspel', 'OutofVocab', 
                        'Noun', 'Past_Tense', 'Verbs', 'Interrogation',
                        'word_count', 'mean_word_length', 'ttr', 'mtld', 'sentiment_score']

# Drop columns not in the list
columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
df.drop(columns=columns_to_drop, inplace=True)
print(len(df))
df.dropna(inplace=True)
print(len(df))

df_e = pd.read_csv("E:/constructive code/constructiveness/data/processed/emotion_scores_nrc.csv")

# List of columns to keep
columns_to_keep = ['fear', 'anger', 'trust', 'surprise' ,
                              'positive', 'negative', 'sadness', 'disgust', 'joy', 'anticipation'] 

# Drop columns not in the list
columns_to_drop = [col for col in df_e.columns if col not in columns_to_keep]
df_e.drop(columns=columns_to_drop, inplace=True)
print(len(df_e))
df_e.fillna(0, inplace=True)
print(len(df_e))

df = pd.concat([df, df_e], axis=1)
df.to_csv("E:/constructive code/constructiveness/data/processed/C3_anonymized_merge_n_s.csv", index=False)
