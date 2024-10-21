import string
import nltk
from nltk.corpus import stopwords
import re
from nltk import pos_tag, word_tokenize, sent_tokenize
import enchant
from nltk.corpus import sentiwordnet as swn
from nrclex import NRCLex
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import random
import os
from tensorflow.keras.models import Sequential
from textblob import TextBlob
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, GRU, Dropout, Reshape

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

from sklearn.base import BaseEstimator, TransformerMixin

class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.expand_dims(X, axis=-1)

# Define the class with feature extraction methods
class FeatureExtractor:
    def __init__(self):
        pass

    def get_sentence_count(self, text):
        sentences = sent_tokenize(text)
        return len(sentences)

    def get_quotes_frequency(self, text, total_sentences):
        double_quotes_count = text.count('“')
        single_quotes_count = text.count("'")
        double_quotes_frequency = double_quotes_count / total_sentences
        single_quotes_frequency = single_quotes_count / total_sentences
        return double_quotes_frequency + single_quotes_frequency

    def get_punctuation_frequency(self, text, total_sentences):
        punctuation_marks = string.punctuation
        punctuation_count = sum(1 for char in text if char in punctuation_marks)
        return punctuation_count / total_sentences

    def get_unique_punctuation_types(self, text):
        punctuation_marks = set(char for char in text if char in string.punctuation)
        return len(punctuation_marks)

    def get_exclamation_frequency(self, text, total_sentences):
        exclamations_count = text.count('!')
        return exclamations_count / total_sentences

    def get_stopword_frequency(self, text, total_sentences):
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        stopwords_count = sum(1 for word in words if word in stop_words)
        return stopwords_count / total_sentences

    def get_camel_case_frequency(self, text, total_sentences):
        camel_case_pattern = re.compile(r'([a-z]+)([A-Z][a-z]*)+')
        matches = camel_case_pattern.findall(text)
        return len(matches) / total_sentences

    # Negations Frequency of no, never, or not
    def get_negations_frequency(self, text, total_sentences):
        negations_pattern = re.compile(r'\b(?:no|never|not)\b', flags=re.IGNORECASE)
        negations_count = len(negations_pattern.findall(text))
        return negations_count / total_sentences

    # Proper Nouns Frequency of POS tags NNP and NNPS
    def get_proper_nouns_frequency(self, text, total_sentences):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        proper_nouns_count = sum(1 for word, pos in pos_tags if pos in ['NNP', 'NNPS'])
        return proper_nouns_count / total_sentences

    # User Mentions Frequency of @
    def get_user_mentions_frequency(self, text, total_sentences):
        words = word_tokenize(text)
        user_mentions_count = sum(1 for word in words if word.startswith('@'))
        return user_mentions_count / total_sentences

    # Hashtags Frequency of # per sentence
    def get_hashtags_frequency(self, text, total_sentences):
        words = word_tokenize(text)
        hashtags_count = sum(1 for word in words if word.startswith('#'))
        return hashtags_count / total_sentences

    # Misspelled Words Frequency of words not considered valid by PyEnchant
    # def get_misspelled_words_frequency(self, text, total_sentences):
    #     words = word_tokenize(text)
    #     english_dict = enchant.Dict("en_US")
    #     misspelled_words_count = sum(1 for word in words if not english_dict.check(word))
    #     return misspelled_words_count / total_sentences

    def get_misspelled_words_frequency(self, text, total_sentences):
        # Create a TextBlob object
        blob = TextBlob(text)
        
        # List to hold misspelled words
        misspelled_words = []

        # Check each word in the text
        for word in blob.words:
            # Identify if the word is misspelled
            if word != word.correct():
                misspelled_words.append(word)
        
        # Calculate frequency of misspelled words
        misspelled_frequency = len(misspelled_words) / total_sentences if total_sentences > 0 else 0
        
        return misspelled_frequency
    # Out of Vocabulary Frequency of words not in the SentiWordNet dictionary
    def get_oov_frequency(self, text, total_sentences):
        words = word_tokenize(text)
        oov_words_count = sum(1 for word in words if not list(swn.senti_synsets(word)))
        return oov_words_count / total_sentences

    # Nouns Frequency of POS tags NNP, NNPS, NN, and NNS
    def get_nouns_frequency(self, text, total_sentences):
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        pos_tags = pos_tag(filtered_words)
        nouns_count = sum(1 for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS'])
        return nouns_count / total_sentences

    # Past Tense Words Frequency of POS tags VBD and VBN per sentence
    def get_past_tense_frequency(self, text, total_sentences):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        past_tense_count = sum(1 for word, pos in pos_tags if pos in ['VBD', 'VBN'])
        return past_tense_count / total_sentences

    def get_verbs_frequency(self, text, total_sentences):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        verbs_count = sum(1 for word, pos in pos_tags if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        return verbs_count / total_sentences

    def get_interrogative_words_frequency(self, text, total_sentences):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        interrogative_words_count = sum(1 for word, pos in pos_tags if pos in ['WRB', 'WDT', 'WP'])
        return interrogative_words_count / total_sentences


    # Word Count Total number of words
    def calculate_word_count(self, text):
        words = word_tokenize(text)
        total_words = len(words)
        return total_words

    # Mean Word Length Average number of characters per word
    def calculate_mean_word_length(self, text):
        words = word_tokenize(text)
        total_characters = sum(len(word) for word in words)
        total_words = len(words)

        if total_words == 0:
            return 0  # Avoid division by zero

        mean_word_length = total_characters / total_words
        return mean_word_length

    # TTR Ratio of unique vocabulary words to overall word count
    def calculate_ttr(self, text):
        words = word_tokenize(text)
        unique_words = set(words)

        if len(words) == 0:
            return 0  # Avoid division by zero

        ttr_ratio = len(unique_words) / len(words)
        return ttr_ratio


    #a TTR value of 0.72 might be indicative of a minimal acceptable level of lexical diversity.
    def calculate_mtld(self, text, threshold=0.72):
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
    def calculate_sentiment_score(self, text):
        words = word_tokenize(text)
        total_sentiment_score = 0.0

        for word in words:
            synsets = list(swn.senti_synsets(word))
            
            if synsets:
                # Consider the first synset for simplicity
                sentiment_score = synsets[0].pos_score() - synsets[0].neg_score()
                total_sentiment_score += sentiment_score

        return total_sentiment_score

    def emotion_nrc(self, text):
        lex = NRCLex(text)
        emotion_scores = lex.affect_frequencies
        return emotion_scores
         

    # Function to process text, find common POS tags, and calculate occurrences per sentence

    def process_text(self, text):
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        pos_counts = {}
        common_pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
                           'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                           'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                           'WDT', 'WP', 'WP$', 'WRB']
        for sentence in sentences:
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            for word, tag in pos_tags:
                if tag in common_pos_tags:
                    pos_counts[tag] = pos_counts.get(tag, 0) + 1
        for key in pos_counts:
            pos_counts[key] /= num_sentences
        return pos_counts

# Step 1: Load the pre-trained model from the .pkl file
def load_model(model_path):
    model = joblib.load(model_path)
    return model
    

# Assuming FeatureExtractor is already defined and loaded
def calculate_features_for_text(single_test_data):
    extractor = FeatureExtractor()
    total_sentences = extractor.get_sentence_count(single_test_data)
    
    features = {
        "Quotes": extractor.get_quotes_frequency(single_test_data, total_sentences),
        "Punc_Freq": extractor.get_punctuation_frequency(single_test_data, total_sentences),
        "Punc_Type": extractor.get_unique_punctuation_types(single_test_data),
        "Exclamation": extractor.get_exclamation_frequency(single_test_data, total_sentences),
        "Stopwords": extractor.get_stopword_frequency(single_test_data, total_sentences),
        "Camelcase": extractor.get_camel_case_frequency(single_test_data, total_sentences),
        "Negation": extractor.get_negations_frequency(single_test_data, total_sentences),
        "Propn": extractor.get_proper_nouns_frequency(single_test_data, total_sentences),
        "Usermention": extractor.get_user_mentions_frequency(single_test_data, total_sentences),
        "Hashtags": extractor.get_hashtags_frequency(single_test_data, total_sentences),
        "Misspel": extractor.get_misspelled_words_frequency(single_test_data, total_sentences),
        "OutofVocab": extractor.get_oov_frequency(single_test_data, total_sentences),
        "Nouns": extractor.get_nouns_frequency(single_test_data, total_sentences),
        "PastTense": extractor.get_past_tense_frequency(single_test_data, total_sentences),
        "Verbs": extractor.get_verbs_frequency(single_test_data, total_sentences),
        "Interrogative": extractor.get_interrogative_words_frequency(single_test_data, total_sentences),
        "Word_Count": extractor.calculate_word_count(single_test_data),
        "MeanWordLength": extractor.calculate_mean_word_length(single_test_data),
        "TTR": extractor.calculate_ttr(single_test_data),
        "MTLD": extractor.calculate_mtld(single_test_data),
        "Sentiment": extractor.calculate_sentiment_score(single_test_data),
        "Emotion": extractor.emotion_nrc(single_test_data),  # Emotion is a dictionary
        "POS_Counts": extractor.process_text(single_test_data)  # POS_Counts is also a dictionary
    }
    
    # Flatten the Emotion dictionary into separate features (e.g., 'fear', 'anger', etc.)
    emotion_dict = features["Emotion"]
    emotion_features = [emotion_dict.get(emotion, 0) for emotion in ["fear", "anger", "anticip", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]]
    
    # Flatten the POS_Counts dictionary into separate features (e.g., 'DT', 'VBZ', etc.)
    pos_counts_dict = features["POS_Counts"]
    pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 
                'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                'WDT', 'WP', 'WP$', 'WRB']
    pos_features = [pos_counts_dict.get(tag, 0) for tag in pos_tags]
    
    # Combine all features into a single list
    all_features = np.array(list(features.values())[:-2] + emotion_features + pos_features).reshape(1, -1)
    
    return all_features

# The remaining code for loading the model and making predictions stays the same


# Step 3: Make predictions with the loaded model
def predict_with_model(model, text_features):
    prediction = model.predict(text_features)
    probability = model.predict_proba(text_features)
    return prediction, probability

def get_sample_comment():
    sample_comments = (["I have 3 daughters, and I told them that Mrs. Clinton lost because she did not have a platform. If she did, she, and her party, did a poor job explaining it to the masses. The only message that I got from her was that Mr. Trump is not fit to be in office and that she wanted to be the first female President. I honestly believe that she lost because she offered no hope, or direction, to the average American. Mr. Trump, with all his shortcomings, at least offered change and some hope. He now has to make it happen or he will be out of power in 4 years.",
                            "Is this a joke? Marie Henein as feminist crusader, advising us what to tell our daughters?? no thanks",
                            "Why don't the NDP also promise 40 acres and a mule? They will never lead this country. Panderers to socialists and unionists.",
                            "In my opinion, criticizing the new generation is not going to solve any problem. If you want to produce children, you should be prepared to pay for their care.",
                            "Simpson is right: it's a political winner and a policy dud - just political smoke and mirrors. Mulcair is power-hungry. He wants Canada to adopt a national childcare model so he can hang on to seats in Quebec, that's all. Years ago I worked with a political strategist working to get a Liberal candidate elected in Conservative Calgary. He actually told his client to talk about national daycare - this was in the early 90's. The Liberal candidate said, `Canada can't afford that!' to which the strategist responded `Just say the words, you don't have to actually do it. It'll be good for votes.' I could barely believe the cynicism, but over the years I've come to realize that's what it is: vote getting and power politics. Same thing here.",
                            "If it happens once in a while to everyone, it's a crime. If it happens disproportionately to a particular demographic, it's more than just a crime. Harper is way out to lunch on this one. This is not an issue the police can solve. It's going to take a society, a country, to do that, and that takes leadership and information."])
    sample_comment = random.choice(sample_comments)
    return sample_comment

def build_lstm_model(units=128, dropout_rate=0.2, input_shape=(1, 1)):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units // 2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_bilstm_model(input_shape, dropout_rate=0.2, units=128):
    model = Sequential()
    #   model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units // 2)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_gru_model(units=128, dropout_rate=0.2, input_shape=(1, 1)):
    model = Sequential()
    model.add(GRU(units=units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_shape):
    model = Sequential()
    
    # Ensure input shape is compatible with Conv1D
    model.add(Reshape((input_shape[0], 1), input_shape=input_shape))
    
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def ensure_2d(probabilities):
    # If the probabilities are 1D (e.g., [0.1, 0.9]), reshape them to 2D
    if len(probabilities.shape) == 1:
        probabilities = np.expand_dims(probabilities, axis=0)
    return probabilities



# Example usage:
if __name__ == "__main__":
    #Adding custom CSS for styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #2ecc71;  /* Set the background color */
            padding: 20px;
            border-radius: 10px;
            width: 600px;
            margin: 0 auto;  /* Center the box */
        }
        .stButton>button {
            background-color:#93c572 ; /* #f0f5fa; */
            color: white;
            border-radius: 5px;
            border: 1px solid green;
            font-size: 16px;
            width: 120px;
            margin: 5px;
        }

        .stTextArea textarea {
            background-color: white;
            color: black;
            border: 3px solid green;
        }
        .header {
            text-align: center;
            color: blue;
        }
        .stApp {
            background-color: white;
        }
        .st-bb {
        color: black;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp a {
        color: black;
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
            color: black;
        }
 
        .stSidebar{
            background-color: #93c572;
            color: black;
            border: 2px solid green;
        }
        # .stSelectbox div[data-baseweb="select"] > div:first-child {
        #     background-color: #FFFFFF;
        #     border-color: #2d408d;
        # } 
        </style>
        """,
        unsafe_allow_html=True
    )
    column_names = [
                'Quotes', 'Punc_Freq', 'Punc_Type', 'Exclamation', 'Stopwords', 'Camelcase',
                 'Negation', 'Propn', 'Usermention', 'Hashtags', 'Misspel', 'OutofVocab', 
                 'Noun', 'Past_Tense', 'Verbs', 'Interrogation',
                'word_count', 'mean_word_length', 'ttr', 'mtld', 
               'fear', 'anger', 'trust', 'surprise' ,
               'positive', 'negative', 'sadness', 'disgust', 'joy', 'anticipation',
               'sentiment_score', 
                'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 
                'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                'WDT', 'WP', 'WP$', 'WRB'
    ] 
    
    # Main container with custom style
    with st.container():
        #st.markdown('<div class="main">', unsafe_allow_html=True)

        # Header
        #st.markdown('<h2 class="header">COMMENT MODERATION</h2>', unsafe_allow_html=True)

        logo_path = "./data/model/logo_unipassau.PNG"
        university_logo = Image.open(logo_path)
        st.image(university_logo, width=700)
        #st.title("Text Analysis for Constructive Comments")
        st.markdown("### University of Passau")
        st.markdown("#### Faculty of Computer Science and Mathematics")
        st.markdown("#### Chair for Computational Rhetoric and Natural Language Processing")
        st.markdown("**Thesis Title:** Exploring Constructiveness in Online News Comments")
        st.markdown("**Submitted by:** Ajay Kesarwani")
        st.markdown("**Examiners:**")
        st.markdown("**1. Prof. Dr. Annette Hautli-Janisz**")
        st.markdown("**2. Prof. Dr. Florian Lemmerich**")
        
        
        st.sidebar.header("Model")
        model_1 = st.sidebar.selectbox("Please select a model:", ["Logistic Regression", "Support Vector Machine", "Random Forest", "K-Nearest Neighbor", "Long Short-Term Memory", "Bi-directional Long-Short Term Memory","Gated-Recurrent Unit", "Convolution Neural Network"])
        # Load the model (provide the path to your .joblib file)
        if model_1 == "Logistic Regression":
            model_path = "/workspaces/Thesis/constructive_final/data/model/Logistic_Regression_best_model.pkl"
        elif model_1 == "Support Vector Machine":
            model_path = "/workspaces/Thesis/constructive_final/data/model/SVM_best_model.pkl"
        elif model_1 == "Random Forest":
            model_path = "/workspaces/Thesis/constructive_final/data/model/Random_Forest_best_model.pkl"
        elif model_1 == "K-Nearest Neighbor":
            model_path = "/workspaces/Thesis/constructive_final/data/model/KNN_best_model.pkl"
        elif model_1 == "Long Short-Term Memory":
            model_path = "/workspaces/Thesis/constructive_final/data/model/LSTM_best_model.pkl"
        elif model_1 == "Bi-directional Long-Short Term Memory":
            model_path = "/workspaces/Thesis/constructive_final/data/model/BiLSTM_best_model.pkl"
        elif model_1 == "Gated-Recurrent Unit":
            model_path = "/workspaces/Thesis/constructive_final/data/model/GRU_best_model.pkl"
        elif model_1 == "Convolution Neural Network":
            model_path = "/workspaces/Thesis/constructive_final/data/model/CNN_best_model.pkl"

        model = load_model(model_path)
        
        
        # User input
        st.subheader("Enter a comment to check constructiveness:")
        # user_input = st.text_area("")
        # if st.button("Get Sample Comment"):
        #     sample_comment = get_sample_comment()
        #     st.write(sample_comment)
        # Initialize the session state for the text area
        # Initialize the session state for the text area
        # if "user_input" not in st.session_state:
        #     st.session_state.user_input = ""

        # # Text area that automatically updates when a sample comment is generated or cleared
        # user_input = st.text_area("Comment:", st.session_state.user_input)

        # # Button to generate a sample comment
        # if st.button("Get Sample Comment"):
        #     sample_comment = get_sample_comment()
        #     st.session_state.user_input = sample_comment  # Update the text area with the sample comment

        # # Button to clear the text area
        # if st.button("Clear"):
        #     st.session_state.user_input = ""  # Clear the text area by resetting session state

        
        # if st.button("Analyze"):
        #     if user_input:
        #         # Feature extraction
        #         features = calculate_features_for_text(user_input)         
        #         text_features = pd.DataFrame(features, columns=column_names)        
        #         # Prediction
        #         prediction, probability = predict_with_model(model, text_features)  

        #         # Determine constructiveness
        #         if prediction == 1:
        #             cons = "CONSTRUCTIVE"
        #         else:
        #             cons = "NON_CONSTRUCTIVE"

        #         predicted_class_prob = probability[0][prediction[0]]

        #         # Display the results
        #         st.write(f"According to our {model_1} model, the comment is likely to be {cons} with a probability of {predicted_class_prob:.2f}")
        
            
        #         # st.subheader("Prediction Results:")
        #         # st.write(f"Predicted Class: {prediction}")
        #         # st.write(f"Prediction Probabilities: {probability}")
        #         print("Prediction Results:")
        #         print(f"Predicted Class: {prediction}")
        #         print(f"Prediction Probabilities: {probability}")
        #     else:
        #         st.error("Please enter some text to analyze.")


        # Button to generate a sample comment (Example Comment at the top)
        if st.button("Get Sample Comment"):
            sample_comment = get_sample_comment()
            st.session_state.user_input = sample_comment  # Update the text area with the sample comment

        # Subheader for the input section
        # st.subheader("Enter a comment to check constructiveness:")

        # Initialize the session state for the text area
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        # Text area for user input or generated comment
        user_input = st.text_area("Comment:", st.session_state.user_input)

        # Display Clear and Submit buttons side by side below the text area
        col2, col1 = st.columns([1, 1])  # Divide the layout into two equal-width columns

        # Clear button
        with col1:
            if st.button("Clear"):
                st.session_state.user_input = ""  # Clear the text area by resetting session state

        # Submit/Analyze button
        with col2:
            if st.button("Analyze"):
                if user_input:
                    # Feature extraction
                    features = calculate_features_for_text(user_input)         
                    text_features = pd.DataFrame(features, columns=column_names)        

                    # Prediction
                    prediction, probability = predict_with_model(model, text_features)  
                    probability = ensure_2d(probability)

                    # Determine constructiveness
                    if prediction[0] == 1:
                        cons = "CONSTRUCTIVE"
                    else:
                        cons = "NON_CONSTRUCTIVE"

                    #predicted_class_prob = probability[0][prediction[0]]
                    # Assuming multi-class or binary classification
                    if isinstance(probability, np.ndarray) and isinstance(prediction, np.ndarray):
                        # Multi-class
                        if len(probability.shape) == 2 and len(prediction) > 0:
                            predicted_class_prob = probability[0][prediction[0]]
                        # Binary classification
                        elif len(probability.shape) == 1:
                            predicted_class_prob = probability[0] if prediction[0] == 1 else 1 - probability[0]
                        else:
                            raise ValueError("Unexpected shape for probability or prediction.")
                    else:
                        raise ValueError("Prediction or probability is not an ndarray.")

                    print(f"Predicted class probability: {predicted_class_prob}")



                    # Display the results
                    st.write(f"According to our {model_1} model, the comment is likely to be {cons} with a probability of {predicted_class_prob:.2f}")

                    # Print prediction results (for debugging or logging purposes)
                    print("Prediction Results:")
                    print(f"Predicted Class: {prediction}")
                    print(f"Prediction Probabilities: {probability}")
                else:
                    st.error("Please enter some text to analyze.")
    
