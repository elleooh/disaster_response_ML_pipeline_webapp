import sys
import re
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - file path for message.csv (e.g. data/message.csv)
    categories_filepath - file path for categories.csv

    OUTPUT
    df - a dataframe with messages and categories combined
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on="id", how="inner")
    return df

def clean_data(df):
    '''
    INPUT
    df - dataframe with messages and categorie combined

    OUTPUT
    df - dataframe with new binary columns based on categories and duplicates removed
    '''

    # split categories into separate category columns
    categories = df.categories.str.split(";", expand = True)

    # select the first row of the categories dataframe
    row = categories[:1].values[0].tolist()

    # extract a list of new column names for categories
    category_colnames = list(map(lambda x: re.match(r'^(.+?)-', x)[0][:-1], row))

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for col in category_colnames:
        # set each value to be the last character of the string
        categories[col] = categories[col].astype(str).str[-1]

        # convert column from string to numeric
        categories[col] = categories[col].astype(int)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # change related to 1 if its value is 2
    df.loc[df['related'] == 2, 'related'] = 1

    # remove duplicates
    df = df[~df.duplicated()]

    # check number of duplicates
    assert df.duplicated().sum()==0,'there are still duplicates'

    return df

def save_data(df, database_filename):
    '''
    INPUT
    df - cleaned dataframe
    database_filename - database name

    OUTPUT
    df saved to the database specified as table etl_messages
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('etl_messages', engine, index=False,  if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '\
        'datasets as the first and second argument respectively, as '\
        'well as the filepath of the database to save the cleaned data '\
        'to as the third argument. \n\nExample: python process_data.py '\
        'disaster_messages.csv disaster_categories.csv '\
        'DisasterResponse.db')

if __name__ == '__main__':
    main()
