from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from my_transformers import TextSelector, NumberSelector

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten, Conv1D, Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, MaxPooling1D
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig




# def split_data(X, y):
#     """
#     """
#     X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
#     X_train, X_valid, y_train, y_valid = train_test_split(X_trainvalid, y_trainvalid, train_size=0.75, random_state=1)

#     print("Number of training examples:", len(y_train))
#     print("Number of validation examples:", len(y_valid))
#     print("Number of test examples:", len(y_test))
    
#     return X_train, y_train, X_valid, y_valid, X_trainvalid, y_trainvalid, X_test, y_test

def text_quality_feats_pipeline():
    '''
    :return:
    '''
    ncaps = Pipeline([
        ('selector', NumberSelector(key='ncaps')),
        ('standard', StandardScaler())
    ])

    noov = Pipeline([
        ('selector', NumberSelector(key='noov')),
        ('standard', StandardScaler())
    ])
    
    readability_score = Pipeline([
        ('selector', NumberSelector(key='readability_score')),
        ('standard', StandardScaler())
    ])

    personal_exp_score = Pipeline([
        ('selector', NumberSelector(key='personal_exp_score')),
        ('standard', StandardScaler())
    ])
    
    text_quality_feats = FeatureUnion([
        ('readability_score', readability_score),
        ('personal_exp_score', personal_exp_score),
        ('ncaps', ncaps),
        ('noov', noov)        
    ])
    # text_quality_feats = FeatureUnion([
    #     ('readability_score', readability_score),
    #     ('personal_exp_score', personal_exp_score)       
    # ])
    return text_quality_feats

def named_entity_feats_pipeline():
    '''
    :return:
    '''
    named_entity_count = Pipeline([
        ('selector', NumberSelector(key='named_entity_count')),
        ('standard', StandardScaler())
    ])

    return named_entity_count

def argumentation_feats_pipeline():
    '''
    #:return:
    '''
    has_conjunctions_and_connectives = Pipeline([
        ('selector', NumberSelector(key='has_conjunctions_and_connectives')),
        ('standard', StandardScaler())
    ])

    has_stance_adverbials = Pipeline([
        ('selector', NumberSelector(key='has_stance_adverbials')),
        ('standard', StandardScaler())
    ])

    has_reasoning_verbs = Pipeline([
        ('selector', NumberSelector(key='has_reasoning_verbs')),
        ('standard', StandardScaler())
    ])

    has_modals = Pipeline([
        ('selector', NumberSelector(key='has_modals')),
        ('standard', StandardScaler())
    ])

    has_shell_nouns = Pipeline([
        ('selector', NumberSelector(key='has_shell_nouns')),
        ('standard', StandardScaler())
    ])
    argumentation_feats = FeatureUnion([
            ('has_conjunctions_and_connectives', has_conjunctions_and_connectives),
            ('has_stance_adverbials', has_stance_adverbials),
            ('has_reasoning_verbs', has_reasoning_verbs),
            ('has_modals', has_modals),
            ('has_shell_nouns', has_shell_nouns)
            ])
    return argumentation_feats



def length_feats_pipeline():
    '''
    #:return:
    '''
    # Length features
    length = Pipeline([
        ('selector', NumberSelector(key='length')),
        ('standard', StandardScaler())
    ])

    average_word_length = Pipeline([
        ('selector', NumberSelector(key='average_word_length')),
        ('standard', StandardScaler())
    ])

    nSents = Pipeline([
        ('selector', NumberSelector(key='nSents')),
        ('standard', StandardScaler())
    ])

    avg_words_per_sent = Pipeline([
        ('selector', NumberSelector(key='avg_words_per_sent')),
        ('standard', StandardScaler())
    ])

    length_feats = FeatureUnion([
        ('length', length),
        ('average_word_length', average_word_length),
        ('nSents', nSents),
        ('avg_words_per_sent', avg_words_per_sent)
    ])

    return length_feats



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
    
import tensorflow as tf

def split_data(X, y, test_size=0.2, valid_size=0.25, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=valid_size, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test

if __name__ == "__main__":
    df1 = pd.read_csv("E:/constructive code/constructiveness/data/processed/C3_anonymized_merge_n_s.csv")
    df2 = pd.read_csv("E:/constructive code/constructiveness/data/processed/pos_tag_nltk_common.csv")
    df2 = df2.drop(columns=['constructive', 'pp_comment_text'])
    df = pd.concat([df1, df2], axis=1)
    print(len(df))

    df.dropna(inplace=True)
    df['constructive_binary'] = df['constructive'].apply(lambda x: 1 if x > 0.5 else 0)

    feature_set = [
                #     'length_feats',
                #    'argumentation_feats',
                #    'named_entity_feats',
                #    'text_quality_feats',
                    'emotion_feats',
                    'stylistic_feats',
                    'complexity_feats',
                    'psychological_feats',
                     'pos_tag_feats'
    ]

    feature_cols = [
    #            'has_conjunctions_and_connectives', 'has_stance_adverbials',
    #            'has_reasoning_verbs', 'has_modals', 'has_shell_nouns', 'length',
    #            'average_word_length', 'readability_score', 'ncaps', 'noov',
    #            'personal_exp_score', 'named_entity_count', 'nSents',
    #            'avg_words_per_sent', 
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


#     feature_set = ['pos_tag_feats']
#     feature_cols = [
#                 'sentiment_score'
#    ]
    
    # X_C3 = df.loc[:, feature_cols]
    # y_C3 = df['constructive_binary']
    # X_C3 = pd.DataFrame(np.random.rand(1000, len(feature_cols)), columns=feature_cols)
    # y_C3 = np.random.randint(0, 2, 1000)

    # Prepare features and target
    X_C3 = df.loc[:, feature_cols]
    y_C3 = df['constructive_binary']

    # Split data
    X_C3_train, y_C3_train, X_C3_valid, y_C3_valid, X_C3_trainvalid, y_C3_trainvalid, X_C3_test, y_C3_test = split_data(X_C3, y_C3)

    feats = build_feature_pipelines_and_unions(feature_set, comments_col='pp_comment_text')
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, feature_cols)
    ])
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Define the regularization parameter
    l2_lambda = 0.001

    # Function to create a simple dense neural network
    def create_model():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(len(feature_cols),), kernel_regularizer=l2(l2_lambda)),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
            Dropout(0.5),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Function to create an LSTM model
    def create_lstm_model():
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(len(feature_cols), 1), kernel_regularizer=l2(l2_lambda)),
            Dropout(0.5),
            LSTM(32, kernel_regularizer=l2(l2_lambda)),
            Dropout(0.5),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Function to create a BiLSTM model
    def create_bilstm_model():
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_lambda)), input_shape=(len(feature_cols), 1)),
            Dropout(0.5),
            Bidirectional(LSTM(32, kernel_regularizer=l2(l2_lambda))),
            Dropout(0.5),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Function to create a CNN model
    # def create_cnn_model():
    #     model = Sequential([
    #         Conv1D(64, kernel_size=3, activation='relu', input_shape=(len(feature_cols), 1), kernel_regularizer=l2(l2_lambda)),
    #         GlobalMaxPooling1D(),
    #         Dropout(0.5),
    #         Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
    #         Dropout(0.5),
    #         Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
    #     ])
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #     return model

    def create_cnn_model():
        sequence_input = Input(shape=(len(feature_cols), 1), dtype='float32')
        x = Conv1D(128, 2, activation='relu', padding='same')(sequence_input)
        x = MaxPooling1D(5, padding='same')(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        x = Conv1D(128, 4, activation='relu', padding='same')(x)
        x = MaxPooling1D(40, padding='same')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(1, activation='sigmoid')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        return model
    # Function to create a GRU model
    def create_gru_model():
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(len(feature_cols), 1), kernel_regularizer=l2(l2_lambda)),
            Dropout(0.5),
            GRU(32, kernel_regularizer=l2(l2_lambda)),
            Dropout(0.5),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Preprocessing function for BERT
    def bert_preprocess(texts, tokenizer):
        return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="tf")

    # Function to evaluate a model
    def evaluate_model(create_model_fn):
        # Wrap the Keras model in a scikit-learn estimator
        model = KerasClassifier(model=create_model_fn, epochs=100, batch_size=32, verbose=0,
                                validation_split=0.2, callbacks=[early_stopping])
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the pipeline
        pipeline.fit(X_C3_train, y_C3_train)
        
        # Evaluate the pipeline
        y_pred = pipeline.predict(X_C3_test)
        accuracy = pipeline.score(X_C3_test, y_C3_test)
        f1 = f1_score(y_C3_test, y_pred)
        report = classification_report(y_C3_test, y_pred, output_dict=True,zero_division=1)
        
        return pipeline, accuracy, f1, report

    # List of model creators
    model_creators = [create_model, create_lstm_model, create_bilstm_model, create_gru_model]

    # Evaluate all models
    results = {}
    pipelines = {}
    for creator in model_creators:
        model_name = creator.__name__
        pipeline, accuracy, f1, report = evaluate_model(creator)
        pipelines[model_name] = pipeline
        results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'report': report
        }

    # Convert results to DataFrame for better display
    results_df = pd.DataFrame.from_dict({name: {'Accuracy': f'{metrics["accuracy"]:.4f}',
                                                'F1 Score': f'{metrics["f1_score"]:.4f}',
                                                'Precision': f'{metrics["report"]["weighted avg"]["precision"]:.4f}',
                                                'Recall': f'{metrics["report"]["weighted avg"]["recall"]:.4f}'}
                                        for name, metrics in results.items()},
                                        orient='index')

    print(results_df)

    # Display full classification reports for each model
    for model_name, metrics in results.items():
        print(f'\nModel: {model_name}')
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'F1 Score: {metrics["f1_score"]:.4f}')
        print('Classification Report:')
        print(classification_report(y_C3_test, pipelines[model_name].predict(X_C3_test),zero_division=1))
        print('-----------------------')