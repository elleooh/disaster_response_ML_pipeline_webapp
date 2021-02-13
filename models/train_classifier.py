import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

def load_data(database_filepath):
    """
    INPUT
    database_filepath: database name where etl_messages was saved in previous step

    OUTPUT
    X: dataframe containing features (i.e.genre and message)
    Y: dataframe containing target (i.e 36 categories)
    category_names: list of category name
    """
    engine = create_engine('sqlite:///{}.db'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM etl_messages", engine)

    category_names = list(set(df.columns)-set(['id', 'message', 'original', 'genre']))
    Y = df[category_names]
    X = pd.concat([pd.get_dummies(df['genre']), df['message']], axis=1)
    return X, Y, category_names

class ItemSelector(BaseEstimator, TransformerMixin):
    """
    https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html
    # Author: Matt Terry <matt.terry@gmail.com>

    For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

def tokenize(text):
    """
    INPUT
    text: text string

    OUTPUT
    tokens: list of word tokens lemmatized and stop words removed
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # lemmatize andremove stop words
    stop_words = stopwords.words("english")
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens

def build_model():
    """
    OUTPUT
    a machine learning pipeline that specifies the feature transformeation and model workflow
    """
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('selector', ItemSelector(key='message')),
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('genre_enc', ItemSelector(key=['direct', 'social', 'news'])),
    ])),
        ('clf', MultiOutputClassifier(XGBClassifier(colsample_bytree=0.6,  max_depth=10))),
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT
    model: model fitted using X_train and Y_train
    X_test: dataframe of features
    Y_test: dataframe of target

    OUTPUT
    print out evaluation metrics for the model (i.e F1, precision, recall, accuracy)
    """
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('Category: ', Y_test.columns[i])
        print(classification_report(pd.DataFrame(y_pred).iloc[:, i], \
            pd.DataFrame(Y_test).iloc[:, i]))

    # Use micro-average as it considers the contributions of each classe to the overall. 
    # It is preferred in our case since the class is imbalanced in our dataset
    print('Overall F1 (micro): ', f1_score(y_pred, Y_test, average='micro'))
    print('Overall precision (micro): ', precision_score(y_pred, Y_test, average='micro'))
    print('Overall recall (micro): ', recall_score(y_pred, Y_test, average='micro'))
    print('Overall accuracy: ', (y_pred == Y_test.values).mean())


def save_model(model, model_filepath):
    """
    INPUT
    model: model fitted using X_train and Y_train
    model_filepath: file path where the pickle model file will be saved

    Save the model as a pickle file
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
