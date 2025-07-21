Constructive Comments 
===========================
This Master Thesis project is to classify the Online News Comments whether it's Constructive or Non-Constructive. 
This study used Linguistic (Stylistic, Complexity and Psychological features and POS Tags from Penn Treebank) and eight basic emotional features(using NRCLex Library) to classify the constructiveness of a given text. 

For this project, we used the Constructive Comments Corpus (C3), which consists of 12,000 comments annotated by crowdworkers, and Yahoo News Annotated Comment Corpus, which consists of 23,383 annotated comments.

- SFU Opinion and Comments Corpus](https://github.com/sfu-discourse-lab/SOCC)
- Constructive Comments Corpus (C3)
    - On Kaggle: [https://www.kaggle.com/mtaboada/c3-constructive-comments-corpus](https://www.kaggle.com/mtaboada/c3-constructive-comments-corpus)

- Yahoo News Annotated Comment Corpus
    - on Github: https://github.com/cnap/ynacc
Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── model            <- Here trained model stored in .pkl file.
    │
    ├── requirements.txt   <- The requirements file for installing the dependent libraries, e.g.
    │                         generated with `pip install -r .\requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── preprocessing           <- Scripts to preprocess the text so it can be used in models
    │   │   └── preprocessor.py
    │   │
    │   ├── features       <- Scripts to turn raw data into fake news detection features
    │   │   └── fnd_n.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── m_logistic.py    
    │   │   ├── m_deeplearning.py
    │   │   └── apps.py    <- Run this file using Streamlit run apps.py to check constructivness comments in real-time.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations across different features
    │       └── feature_correaltion.py
    │
    └

You can also test any comment, whether it's constructive or non-constructive, using the below hosted link:
https://ajay-constructiveness.streamlit.app/
 
