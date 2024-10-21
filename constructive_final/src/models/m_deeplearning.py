import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report, roc_curve, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, GRU, Dropout, Reshape
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from my_transformers import TextSelector, NumberSelector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from scikeras.wrappers import KerasClassifier
import joblib

def stylistic_feats_pipeline():
    #Quotes	Punc_Freq	Punc_Type	Exclamation	Stopwords	Camelcase	Negation	Propn	Usermention	Hashtags	Misspel	OutofVocab	Noun	Past_Tense	Verbs	Interrogation	

    '''
    #:return:
    '''
    # Length features
    Quotes = Pipeline([
        ('selector', NumberSelector(key='Quotes')),
        ('standard', StandardScaler())
    ])

    Punc_Freq = Pipeline([
        ('selector', NumberSelector(key='Punc_Freq')),
        ('standard', StandardScaler())
    ])

    Punc_Type = Pipeline([
        ('selector', NumberSelector(key='Punc_Type')),
        ('standard', StandardScaler())
    ])

    Exclamation = Pipeline([
        ('selector', NumberSelector(key='Exclamation')),
        ('standard', StandardScaler())
    ])

    Stopwords = Pipeline([
        ('selector', NumberSelector(key='Stopwords')),
        ('standard', StandardScaler())
    ])

    Camelcase = Pipeline([
        ('selector', NumberSelector(key='Camelcase')),
        ('standard', StandardScaler())
    ])

    Negation = Pipeline([
        ('selector', NumberSelector(key='Negation')),
        ('standard', StandardScaler())
    ])

    Propn = Pipeline([
        ('selector', NumberSelector(key='Propn')),
        ('standard', StandardScaler())
    ])

    Usermention = Pipeline([
        ('selector', NumberSelector(key='Usermention')),
        ('standard', StandardScaler())
    ])

    Hashtags = Pipeline([
        ('selector', NumberSelector(key='Hashtags')),
        ('standard', StandardScaler())
    ])

    Misspel = Pipeline([
        ('selector', NumberSelector(key='Misspel')),
        ('standard', StandardScaler())
    ])

    OutofVocab = Pipeline([
        ('selector', NumberSelector(key='OutofVocab')),
        ('standard', StandardScaler())
    ])

    Noun = Pipeline([
        ('selector', NumberSelector(key='Noun')),
        ('standard', StandardScaler())
    ])

    Past_Tense = Pipeline([
        ('selector', NumberSelector(key='Past_Tense')),
        ('standard', StandardScaler())
    ])

    Verbs = Pipeline([
        ('selector', NumberSelector(key='Verbs')),
        ('standard', StandardScaler())
    ])

    Interrogation = Pipeline([
        ('selector', NumberSelector(key='Interrogation')),
        ('standard', StandardScaler())
    ])
    stylistic_feats = FeatureUnion([
        ('Quotes', Quotes),
        ('Punc_Freq', Punc_Freq),
        ('Punc_Type', Punc_Type),
        ('Exclamation', Exclamation),
        ('Stopwords', Stopwords),
        ('Camelcase', Camelcase),
        ('Negation', Negation),
        ('Propn', Propn),
        ('Usermention', Usermention),
        ('Hashtags', Hashtags),
        ('Misspel', Misspel),
        ('OutofVocab', OutofVocab),
        ('Noun', Noun),
        ('Past_Tense', Past_Tense),
        ('Verbs', Verbs),
        ('Interrogation', Interrogation)
    ])

    return stylistic_feats

def complexity_feats_pipeline():
    #word_count	mean_word_length	ttr	mtld
    '''
    #:return:
    '''
    # Length features
    word_count = Pipeline([
        ('selector', NumberSelector(key='word_count')),
        ('standard', StandardScaler())
    ])

    mean_word_length = Pipeline([
        ('selector', NumberSelector(key='mean_word_length')),
        ('standard', StandardScaler())
    ])

    ttr = Pipeline([
        ('selector', NumberSelector(key='ttr')),
        ('standard', StandardScaler())
    ])

    mtld = Pipeline([
        ('selector', NumberSelector(key='mtld')),
        ('standard', StandardScaler())
    ])

    complexity_feats = FeatureUnion([
        ('word_count', word_count),
        ('mean_word_length', mean_word_length),
        ('ttr', ttr),
        ('mtld', mtld)
    ])

    return complexity_feats
    

def psychological_feats_pipeline():
    #sentiment_score
    '''
    #:return:
    '''
    # Length features
    sentiment_score = Pipeline([
        ('selector', NumberSelector(key='sentiment_score')),
        ('standard', StandardScaler())
    ])

    psychological_feats = FeatureUnion([
        ('sentiment_score', sentiment_score)
    ])

    return psychological_feats

def emotion_feats_pipeline():
    #fear     anger    trust  surprise  positive  negative   sadness   disgust       joy  anticipation
    '''
    #:return:
    '''
    # Length features
    anticipation = Pipeline([
        ('selector', NumberSelector(key='anticipation')),
        ('standard', StandardScaler())
    ])

    joy = Pipeline([
        ('selector', NumberSelector(key='joy')),
        ('standard', StandardScaler())
    ])

    disgust = Pipeline([
        ('selector', NumberSelector(key='disgust')),
        ('standard', StandardScaler())
    ])

    sadness = Pipeline([
        ('selector', NumberSelector(key='sadness')),
        ('standard', StandardScaler())
    ])

    surprise = Pipeline([
        ('selector', NumberSelector(key='surprise')),
        ('standard', StandardScaler())
    ])

    trust = Pipeline([
        ('selector', NumberSelector(key='trust')),
        ('standard', StandardScaler())
    ])

    anger = Pipeline([
        ('selector', NumberSelector(key='anger')),
        ('standard', StandardScaler())
    ])

    fear = Pipeline([
        ('selector', NumberSelector(key='fear')),
        ('standard', StandardScaler())
    ])

    positive = Pipeline([
        ('selector', NumberSelector(key='positive')),
        ('standard', StandardScaler())
    ])

    negative = Pipeline([
        ('selector', NumberSelector(key='negative')),
        ('standard', StandardScaler())
    ])

    emotion_feats = FeatureUnion([
        ('anticipation', anticipation),
        ('joy', joy),
        ('disgust', disgust),
        ('sadness', sadness),
        ('surprise', surprise),
        ('trust', trust),
        ('anger', anger),
        ('fear', fear),
        ('positive', positive),
        ('negative', negative)
    ])

    return emotion_feats
    
def pos_tag_feats_pipeline():
    '''
    :return:
    '''
    # Defining individual pipelines for each POS tag
    CC = Pipeline([
        ('selector', NumberSelector(key='CC')),
        ('standard', StandardScaler())
    ])
    
    CD = Pipeline([
        ('selector', NumberSelector(key='CD')),
        ('standard', StandardScaler())
    ])
    
    DT = Pipeline([
        ('selector', NumberSelector(key='DT')),
        ('standard', StandardScaler())
    ])
    
    EX = Pipeline([
        ('selector', NumberSelector(key='EX')),
        ('standard', StandardScaler())
    ])
    
    FW = Pipeline([
        ('selector', NumberSelector(key='FW')),
        ('standard', StandardScaler())
    ])
    
    IN = Pipeline([
        ('selector', NumberSelector(key='IN')),
        ('standard', StandardScaler())
    ])
    
    JJ = Pipeline([
        ('selector', NumberSelector(key='JJ')),
        ('standard', StandardScaler())
    ])
    
    JJR = Pipeline([
        ('selector', NumberSelector(key='JJR')),
        ('standard', StandardScaler())
    ])
    
    JJS = Pipeline([
        ('selector', NumberSelector(key='JJS')),
        ('standard', StandardScaler())
    ])
    
    LS = Pipeline([
        ('selector', NumberSelector(key='LS')),
        ('standard', StandardScaler())
    ])
    
    MD = Pipeline([
        ('selector', NumberSelector(key='MD')),
        ('standard', StandardScaler())
    ])
    
    NN = Pipeline([
        ('selector', NumberSelector(key='NN')),
        ('standard', StandardScaler())
    ])
    
    NNS = Pipeline([
        ('selector', NumberSelector(key='NNS')),
        ('standard', StandardScaler())
    ])
    
    NNP = Pipeline([
        ('selector', NumberSelector(key='NNP')),
        ('standard', StandardScaler())
    ])
    
    NNPS = Pipeline([
        ('selector', NumberSelector(key='NNPS')),
        ('standard', StandardScaler())
    ])
    
    PDT = Pipeline([
        ('selector', NumberSelector(key='PDT')),
        ('standard', StandardScaler())
    ])
    
    POS = Pipeline([
        ('selector', NumberSelector(key='POS')),
        ('standard', StandardScaler())
    ])
    
    PRP = Pipeline([
        ('selector', NumberSelector(key='PRP')),
        ('standard', StandardScaler())
    ])
    
    PRP_ = Pipeline([
        ('selector', NumberSelector(key='PRP$')),
        ('standard', StandardScaler())
    ])
    
    RB = Pipeline([
        ('selector', NumberSelector(key='RB')),
        ('standard', StandardScaler())
    ])
    
    RBR = Pipeline([
        ('selector', NumberSelector(key='RBR')),
        ('standard', StandardScaler())
    ])
    
    RBS = Pipeline([
        ('selector', NumberSelector(key='RBS')),
        ('standard', StandardScaler())
    ])
    
    RP = Pipeline([
        ('selector', NumberSelector(key='RP')),
        ('standard', StandardScaler())
    ])
    
    SYM = Pipeline([
        ('selector', NumberSelector(key='SYM')),
        ('standard', StandardScaler())
    ])
    
    TO = Pipeline([
        ('selector', NumberSelector(key='TO')),
        ('standard', StandardScaler())
    ])
    
    UH = Pipeline([
        ('selector', NumberSelector(key='UH')),
        ('standard', StandardScaler())
    ])
    
    VB = Pipeline([
        ('selector', NumberSelector(key='VB')),
        ('standard', StandardScaler())
    ])
    
    VBD = Pipeline([
        ('selector', NumberSelector(key='VBD')),
        ('standard', StandardScaler())
    ])
    
    VBG = Pipeline([
        ('selector', NumberSelector(key='VBG')),
        ('standard', StandardScaler())
    ])
    
    VBN = Pipeline([
        ('selector', NumberSelector(key='VBN')),
        ('standard', StandardScaler())
    ])
    
    VBP = Pipeline([
        ('selector', NumberSelector(key='VBP')),
        ('standard', StandardScaler())
    ])
    
    VBZ = Pipeline([
        ('selector', NumberSelector(key='VBZ')),
        ('standard', StandardScaler())
    ])
    
    WDT = Pipeline([
        ('selector', NumberSelector(key='WDT')),
        ('standard', StandardScaler())
    ])
    
    WP = Pipeline([
        ('selector', NumberSelector(key='WP')),
        ('standard', StandardScaler())
    ])
    
    WP_ = Pipeline([
        ('selector', NumberSelector(key='WP$')),
        ('standard', StandardScaler())
    ])
    
    WRB = Pipeline([
        ('selector', NumberSelector(key='WRB')),
        ('standard', StandardScaler())
    ])

    # Combining all POS tag pipelines into a FeatureUnion
    pos_tag_feats = FeatureUnion([
        ('CC', CC),
        ('CD', CD),
        ('DT', DT),
        ('EX', EX),
        ('FW', FW),
        ('IN', IN),
        ('JJ', JJ),
        ('JJR', JJR),
        ('JJS', JJS),
        ('LS', LS),
        ('MD', MD),
        ('NN', NN),
        ('NNS', NNS),
        ('NNP', NNP),
        ('NNPS', NNPS),
        ('PDT', PDT),
        ('POS', POS),
        ('PRP', PRP),
        ('PRP$', PRP_),
        ('RB', RB),
        ('RBR', RBR),
        ('RBS', RBS),
        ('RP', RP),
        ('SYM', SYM),
        ('TO', TO),
        ('UH', UH),
        ('VB', VB),
        ('VBD', VBD),
        ('VBG', VBG),
        ('VBN', VBN),
        ('VBP', VBP),
        ('VBZ', VBZ),
        ('WDT', WDT),
        ('WP', WP),
        ('WP$', WP_),
        ('WRB', WRB)
    ])

    return pos_tag_feats

def build_feature_pipelines_and_unions(feature_set = ['ngram_feats',
                                                      'tfidf_feats',
                                                      'pos_feats',
                                                      'length_feats',
                                                      'emotion_feats',
                                                      'stylistic_feats',
                                                      'complexity_feats',
                                                      'psychological_feats',
                                                      'pos_tag_feats',
                                                      'argumentation_feats',
                                                      'COMMENTIQ_feats',
                                                     'punctuation_and_typos_feats', 
                                                     'named_entity_feats',
                                                     'constructiveness_chars_feats',
                                                     'non_constructiveness_chars_feats',
                                                     'toxicity_chars_feats', 
                                                     'perspective_content_value_feats', 
                                                     'perspective_aggressiveness_feats',
                                                     'perspecitive_toxicity_feats'], 
                                       comments_col = 'pp_comment_text'):
                                       
    '''
    :return: 
    '''
    print(f"Features Set", feature_set)

    # length_feats = length_feats_pipeline()
    # argumentation_feats = argumentation_feats_pipeline()
    # named_entity_feats = named_entity_feats_pipeline()
    # text_quality_feats = text_quality_feats_pipeline()
    emotion_feats = emotion_feats_pipeline()
    stylistic_feats = stylistic_feats_pipeline()
    complexity_feats = complexity_feats_pipeline()
    psychological_feats = psychological_feats_pipeline()
    pos_tag_feats = pos_tag_feats_pipeline()
    
    feat_sets_dict = {
                    #   'length_feats': length_feats,
                    #   'argumentation_feats': argumentation_feats,
                    #   'named_entity_feats': named_entity_feats,
                    #   'text_quality_feats': text_quality_feats,
                     'emotion_feats': emotion_feats,
                      'stylistic_feats': stylistic_feats,
                      'complexity_feats': complexity_feats,
                      'psychological_feats': psychological_feats,
                      'pos_tag_feats' : pos_tag_feats
                     }
    # feat_sets_dict = {
    #                   'emotion_feats': emotion_feats
                      
    #                  }
    
    feat_tuples = [(feat, feat_sets_dict.get(feat,0)) for feat in feature_set]
    feats = FeatureUnion(feat_tuples)
    return feats

def split_data(X, y, test_size=0.2, valid_size=0.25, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=valid_size, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test

from sklearn.base import BaseEstimator, TransformerMixin

class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.expand_dims(X, axis=-1)

# def evaluate_model(model_name, model_pipeline):
#     # Train the pipeline
#     model_pipeline.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred_prob = model_pipeline.predict(X_test).ravel()
#     y_pred = (y_pred_prob > 0.5).astype(int)

#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred_prob)
#     precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
#     avg_precision = average_precision_score(y_test, y_pred_prob)
#     conf_matrix = confusion_matrix(y_test, y_pred)

#     print(f"\n{model_name} Performance Metrics:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"ROC AUC: {roc_auc:.4f}")
#     print(f"Average Precision: {avg_precision:.4f}")
#     print(f"Classification Report:")
#     print(classification_report(y_test, y_pred))

#     # Plot ROC Curve
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#     plt.figure(figsize=(10, 8))
#     plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve for {model_name}')
#     plt.legend()
#     plt.show()

#     # Plot Precision-Recall Curve
#     plt.figure(figsize=(10, 8))
#     plt.plot(recall, precision, label=f"{model_name} (AP = {avg_precision:.2f})")
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title(f'Precision-Recall Curve for {model_name}')
#     plt.legend()
#     plt.show()

#     # Plot Confusion Matrix
#     ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
#     plt.title(f'Confusion Matrix for {model_name}')
#     plt.show()

#     return {
#         'accuracy': accuracy,
#         'f1': f1,
#         'roc_auc': roc_auc,
#         'average_precision': avg_precision,
#         'conf_matrix': conf_matrix
#     }

def evaluate_model(model_name, model_pipeline, X_test, y_test):
    y_pred_prob = model_pipeline.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    avg_precision = average_precision_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # print(f"\n{model_name} Performance Metrics:")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"ROC AUC: {roc_auc:.4f}")
    # print(f"Average Precision: {avg_precision:.4f}")
    # print(f"Classification Report:")
    # print(classification_report(y_test, y_pred))

    # # Plot ROC Curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    # plt.figure(figsize=(10, 8))
    # plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'ROC Curve for POS Tag Features for {model_name}')
    # plt.legend()
    # output_path = f'ROC Curve for POS Tag Features for {model_name}.png'
    # plt.savefig(output_path)
    # plt.close()


    # # Plot Precision-Recall Curve
    # plt.figure(figsize=(10, 8))
    # plt.plot(recall, precision, label=f"{model_name} (AP = {avg_precision:.2f})")
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve for POS Tag Features for {model_name}')
    # plt.legend()
    # output_path = f'Precision-Recall Curve for POS Tag Features for {model_name}.png'
    # plt.savefig(output_path)
    # plt.close()

    # # Plot Confusion Matrix
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
    # plt.title(f'Confusion Matrix for POS Tag Features for {model_name}')
    # output_path = f'Confusion Matrix for POS Tag Features for {model_name}.png'
    # plt.savefig(output_path)
    # plt.close()

    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'conf_matrix': conf_matrix
    }

# def build_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(64))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

def build_lstm_model(units=128, dropout_rate=0.2, input_shape=(1, 1)):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units // 2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# def build_bilstm_model(input_shape):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(Bidirectional(LSTM(64)))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# Define the BiLSTM model function
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

# def build_gru_model(input_shape):
#     model = Sequential()
#     model.add(GRU(128, input_shape=input_shape, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(GRU(64))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# Define your GRU model function
def build_gru_model(units=128, dropout_rate=0.2, input_shape=(1, 1)):
    model = Sequential()
    model.add(GRU(units=units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    #C3_anonymized_merge_n_s
    #yahoo_data_merge_n_s
    #pos_tag_nltk_common_y
    #df1 = pd.read_csv("./data/processed/yahoo_data_merge_n_s.csv")
    #df2 = pd.read_csv("./data/processed/pos_tag_nltk_common_y.csv")
    # df2 = df2.drop(columns=['constructive', 'pp_comment_text'])

    #df2.dropna(subset=['constructiveclass'], inplace=True)
    #print("Length of df after dropping NA:", len(df2))
    #df2 = df2.drop(columns=['constructiveclass', 'text'])


    #print(len(df2))
    
    #df = pd.concat([df1, df2], axis=1)
    #df.dropna(inplace=True)
    #print(len(df))
    #df['constructive_binary'] = df['constructive'].apply(lambda x: 1 if x > 0.5 else 0)
    df1 = pd.read_csv("./data/processed/C3_anonymized_merge_n_s.csv")
    df2 = pd.read_csv("./data/processed/pos_tag_nltk_common.csv")
    
    # df2.dropna(subset=['constructiveclass'], inplace=True)
    df2 = df2.drop(columns=['constructive', 'pp_comment_text'])

    df = pd.concat([df1, df2], axis=1)
    df.dropna(inplace=True)
    df['constructive_binary'] = df['constructive'].apply(lambda x: 1 if x > 0.5 else 0)
    print(len(df))
    #df = df.head(10)
    print(len(df))

    df['constructive_binary'] = df['constructive'].apply(lambda x: 1 if x > 0.5 else 0)
    counts = df['constructive_binary'].value_counts()


    feature_set = [
                     'stylistic_feats',
                  'complexity_feats',
                    'psychological_feats',
                   'emotion_feats',
                   'pos_tag_feats'
    ]

    feature_cols = [
                'fear', 'anger', 'trust', 'surprise' ,
               'positive', 'negative', 'sadness', 'disgust', 'joy', 'anticipation',
                'Quotes', 'Punc_Freq', 'Punc_Type', 'Exclamation', 'Stopwords', 'Camelcase',
                 'Negation', 'Propn', 'Usermention', 'Hashtags', 'Misspel', 'OutofVocab', 
                 'Noun', 'Past_Tense', 'Verbs', 'Interrogation',
               'word_count', 'mean_word_length', 'ttr', 'mtld', 
               'sentiment_score', 
                'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 
                'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                'WDT', 'WP', 'WP$', 'WRB'
    ] 

    
    X_C3 = df.loc[:, feature_cols].values
    y_C3 = df['constructive_binary'].values

    # X_C3_train, y_C3_train, X_C3_valid, y_C3_valid, X_C3_trainvalid, y_C3_trainvalid, X_C3_test, y_C3_test = split_data(X_C3, y_C3)
    X_train, X_test, y_train, y_test = train_test_split(X_C3, y_C3, test_size=0.2, random_state=42)

    
    feats = build_feature_pipelines_and_unions(feature_set, comments_col='text')

    
    X = df[feature_cols]
    y = df['constructive_binary']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    lstm_model = KerasClassifier(
        #model = build_lstm_model,
        #model=build_bilstm_model,
        model = build_gru_model,
        units=128,
        dropout_rate=0.2,
        input_shape=(X_train.shape[1], 1),
        verbose=2,
        callbacks=[early_stopping]
    )

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'model__units': [32, 64, 128],
        'model__dropout_rate': [0.2, 0.3],
        'model__batch_size': [16, 32, 64],
        'model__epochs': [30]
    }

    lstm_pipeline = Pipeline([
        ('reshape', ReshapeTransformer()),
        ('model', lstm_model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #grid = RandomizedSearchCV(estimator=lstm_pipeline, param_distributions=param_grid, cv=cv, scoring='accuracy', verbose=2, n_jobs=-1)
    grid = RandomizedSearchCV(estimator=lstm_pipeline, param_distributions=param_grid, cv=cv, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train,model__validation_split=0.2, model__callbacks=[early_stopping])

    # Summarize the results
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

    best_model = grid_result.best_estimator_
    best_model.named_steps['model'].model_.summary()
    results = evaluate_model("BiLSTM", best_model, X_test, y_test)
    

    filename = "GRU_best_model.pkl"
    joblib.dump(best_model, filename)


    print("Evaluation results:", results)

    print(results)
