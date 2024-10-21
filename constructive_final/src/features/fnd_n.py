import string
import nltk
from nltk.corpus import stopwords
import re
from nltk import pos_tag, word_tokenize
import enchant
from nltk.corpus import sentiwordnet as swn
import pandas as pd
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')


def get_sentence_count(text):
    sentences = sent_tokenize(text)
    return len(sentences)

def get_quotes_frequency(text, total_sentences):
    double_quotes_count = text.count('“')
    single_quotes_count = text.count("'")
    double_quotes_frequency = double_quotes_count / total_sentences
    single_quotes_frequency = single_quotes_count / total_sentences
    return double_quotes_frequency + single_quotes_frequency

def get_punctuation_frequency(text, total_sentences):
    punctuation_marks = string.punctuation
    punctuation_count = sum(1 for char in text if char in punctuation_marks)
    return punctuation_count / total_sentences

def get_unique_punctuation_types(text):
    punctuation_marks = set(char for char in text if char in string.punctuation)
    return len(punctuation_marks)

def get_exclamation_frequency(text, total_sentences):
    exclamations_count = text.count('!')
    return exclamations_count / total_sentences

def get_stopword_frequency(text, total_sentences):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    stopwords_count = sum(1 for word in words if word in stop_words)
    return stopwords_count / total_sentences

def get_camel_case_frequency(text, total_sentences):
    camel_case_pattern = re.compile(r'([a-z]+)([A-Z][a-z]*)+')
    matches = camel_case_pattern.findall(text)
    return len(matches) / total_sentences

# Negations Frequency of no, never, or not
def get_negations_frequency(text, total_sentences):
    negations_pattern = re.compile(r'\b(?:no|never|not)\b', flags=re.IGNORECASE)
    negations_count = len(negations_pattern.findall(text))
    return negations_count / total_sentences

# Proper Nouns Frequency of POS tags NNP and NNPS
def get_proper_nouns_frequency(text, total_sentences):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    proper_nouns_count = sum(1 for word, pos in pos_tags if pos in ['NNP', 'NNPS'])
    return proper_nouns_count / total_sentences

# User Mentions Frequency of @
def get_user_mentions_frequency(text, total_sentences):
    words = word_tokenize(text)
    user_mentions_count = sum(1 for word in words if word.startswith('@'))
    return user_mentions_count / total_sentences

# Hashtags Frequency of # per sentence
def get_hashtags_frequency(text, total_sentences):
    words = word_tokenize(text)
    hashtags_count = sum(1 for word in words if word.startswith('#'))
    return hashtags_count / total_sentences

# Misspelled Words Frequency of words not considered valid by PyEnchant
def get_misspelled_words_frequency(text, total_sentences):
    words = word_tokenize(text)
    english_dict = enchant.Dict("en_US")
    misspelled_words_count = sum(1 for word in words if not english_dict.check(word))
    return misspelled_words_count / total_sentences

# Out of Vocabulary Frequency of words not in the SentiWordNet dictionary
def get_oov_frequency(text, total_sentences):
    words = word_tokenize(text)
    oov_words_count = sum(1 for word in words if not list(swn.senti_synsets(word)))
    return oov_words_count / total_sentences

# Nouns Frequency of POS tags NNP, NNPS, NN, and NNS
def get_nouns_frequency(text, total_sentences):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    pos_tags = pos_tag(filtered_words)
    nouns_count = sum(1 for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS'])
    return nouns_count / total_sentences

# Past Tense Words Frequency of POS tags VBD and VBN per sentence
def get_past_tense_frequency(text, total_sentences):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    past_tense_count = sum(1 for word, pos in pos_tags if pos in ['VBD', 'VBN'])
    return past_tense_count / total_sentences

def get_verbs_frequency(text, total_sentences):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    verbs_count = sum(1 for word, pos in pos_tags if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
    return verbs_count / total_sentences

def get_interrogative_words_frequency(text, total_sentences):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    interrogative_words_count = sum(1 for word, pos in pos_tags if pos in ['WRB', 'WDT', 'WP'])
    return interrogative_words_count / total_sentences


# Word Count Total number of words
def calculate_word_count(text):
    words = word_tokenize(text)
    total_words = len(words)
    return total_words

# Mean Word Length Average number of characters per word
def calculate_mean_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    total_words = len(words)

    if total_words == 0:
        return 0  # Avoid division by zero

    mean_word_length = total_characters / total_words
    return mean_word_length

# TTR Ratio of unique vocabulary words to overall word count
def calculate_ttr(text):
    words = word_tokenize(text)
    unique_words = set(words)

    if len(words) == 0:
        return 0  # Avoid division by zero

    ttr_ratio = len(unique_words) / len(words)
    return ttr_ratio


#a TTR value of 0.72 might be indicative of a minimal acceptable level of lexical diversity.
def calculate_mtld(text, threshold=0.72):
    words = word_tokenize(text)
    total_words = len(words)
    
    if total_words == 0:
        return 0  # Avoid division by zero
    
    unique_words = set(words)
    ttr = len(unique_words) / total_words
    
    if ttr == 1:
        return 0  # Avoid division by zero
    
    if ttr >= threshold:
        return 1 / (1 - ttr)
    
    segments = [words[i:i+100] for i in range(0, total_words, 100)]
    ttr_sum = sum(len(set(segment)) / len(segment) for segment in segments)
    mtld = total_words / ttr_sum
    
    return mtld

# Sentiment Score Summed SentiWordNet scores for all available vocabulary words
def calculate_sentiment_score(text):
    words = word_tokenize(text)
    total_sentiment_score = 0.0

    for word in words:
        synsets = list(swn.senti_synsets(word))
        
        if synsets:
            # Consider the first synset for simplicity
            sentiment_score = synsets[0].pos_score() - synsets[0].neg_score()
            total_sentiment_score += sentiment_score

    return total_sentiment_score


# Read the CSV file into a DataFrame
df = pd.read_csv("./data/processed/yahoo_data.csv")
#df = pd.read_csv("./data/processed/C3_anonymized.csv")
# print(len(df))
# df.dropna(subset=['constructiveclass'], inplace=True)

# Initialize columns for each feature
columns = [
    "Quotes", "Punc_Freq", "Punc_Type", "Exclamation", "Stopwords",
    "Camelcase", "Negation", "Propn", "Usermention", "Hashtags",
    "Misspel", "OutofVocab", "Noun", "Past_Tense", "Verbs",
    "Interrogation", "word_count", "mean_word_length", "ttr",
    "mtld", "sentiment_score"
]
for col in columns:
    df[col] = 0.0

#print("init")

# Iterate over rows and calculate features
for i in range(len(df)):
    comments = df["text"][i]
    #print(i)
    if comments:
        total_sentences = get_sentence_count(comments)
        if total_sentences > 0:
            df.at[i, "Quotes"] = get_quotes_frequency(comments,total_sentences)
            df.at[i, "Punc_Freq"] = get_punctuation_frequency(comments,total_sentences)
            df.at[i, "Punc_Type"] = get_unique_punctuation_types(comments)
            df.at[i, "Exclamation"] = get_exclamation_frequency(comments,total_sentences)
            df.at[i, "Stopwords"] = get_stopword_frequency(comments,total_sentences)
            df.at[i, "Camelcase"] = get_camel_case_frequency(comments,total_sentences)
            df.at[i, "Negation"] = get_negations_frequency(comments,total_sentences)
            df.at[i, "Propn"] = get_proper_nouns_frequency(comments,total_sentences)
            df.at[i, "Usermention"] = get_user_mentions_frequency(comments,total_sentences)
            df.at[i, "Hashtags"] = get_hashtags_frequency(comments,total_sentences)
            df.at[i, "Misspel"] = get_misspelled_words_frequency(comments,total_sentences)  
            df.at[i, "OutofVocab"] = get_oov_frequency(comments,total_sentences)
            df.at[i, "Noun"] = get_nouns_frequency(comments,total_sentences)
            df.at[i, "Past_Tense"] = get_past_tense_frequency(comments,total_sentences)
            df.at[i, "Verbs"] = get_verbs_frequency(comments,total_sentences)
            df.at[i, "Interrogation"] = get_interrogative_words_frequency(comments,total_sentences)

            df.at[i, "word_count"] = calculate_word_count(comments)
            df.at[i, "mean_word_length"] = calculate_mean_word_length(comments)
            df.at[i, "ttr"] = calculate_ttr(comments)
            df.at[i, "mtld"] = calculate_mtld(comments)

            df.at[i, "sentiment_score"] = calculate_sentiment_score(comments)

# Display the DataFrame with added features
print(df.head())


df.to_csv("./data/processed/yahoo_data_fnd_n_s.csv",  index=False)
#df.to_csv("./data/processed/C3_anonymized_fnd_n_s.csv",  index=False)
