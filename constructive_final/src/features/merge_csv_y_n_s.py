import pandas as pd

# Load the Yahoo data
df = pd.read_csv("E:/constructive code/constructiveness/data/processed/yahoo_data_fnd_n_s.csv")
print("Initial length of df:", len(df))

# Load the emotion scores data
df_e = pd.read_csv("E:/constructive code/constructiveness/data/processed/emotion_scores_yahoo_nrc.csv")
print("Initial length of df_e:", len(df_e))

# Drop NA values based on 'constructiveclass'
df.dropna(subset=['constructiveclass'], inplace=True)
print("Length of df after dropping NA:", len(df))

# Create 'constructive' column
df['constructive'] = df['constructiveclass'].apply(lambda x: 1.0 if x == "Constructive" else 0.0)
print("Length of df after creating 'constructive':", len(df))

# Columns to keep
columns_to_keep = [
    'constructive', 'Quotes', 'Punc_Freq', 'Punc_Type', 'Exclamation', 'Stopwords', 'Camelcase', 'Negation', 
    'Propn', 'Usermention', 'Hashtags', 'Misspel', 'OutofVocab', 'Noun', 'Past_Tense', 'Verbs', 
    'Interrogation', 'word_count', 'mean_word_length', 'ttr', 'mtld', 'sentiment_score'
]

# Drop columns not in the list
columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
df.drop(columns=columns_to_drop, inplace=True)
print("Length of df after dropping columns:", len(df))

# Drop any remaining NA values
df.dropna(inplace=True)
print("Length of df after dropping remaining NA:", len(df))

# Columns to keep in emotion scores
columns_to_keep = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy', 'anticipation'] 

# Drop columns not in the list
columns_to_drop = [col for col in df_e.columns if col not in columns_to_keep]
df_e.drop(columns=columns_to_drop, inplace=True)
print("Length of df_e after dropping columns:", len(df_e))

# Fill NA values with 0
df_e.fillna(0, inplace=True)
print("Length of df_e after filling NA:", len(df_e))

# Reset indices of both DataFrames before concatenation
df_e = df_e.reset_index(drop=True)
df = df.reset_index(drop=True)

# Concatenate the DataFrames
df = pd.concat([df, df_e], axis=1)

# Save the merged DataFrame
df.to_csv("E:/constructive code/constructiveness/data/processed/yahoo_data_merge_n_s.csv", index=False)

# Print the head of the DataFrame to verify
print(df.head())

# Print the length of the DataFrame
print("Final length of df:", len(df))
