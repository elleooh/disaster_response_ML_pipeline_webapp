# Disaster Response ML Pipeline & Webapp

This project builds a ETL and a machine learning pipeline utilizing natural language processing to categorize emergency messages based on the needs communicated by the sender during disaster events. 

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Data](#data)
4. [File Descriptions](#files)
5. [Instructions](#instructions)

## Installation <a name="installation"></a>

Python 3.0 is required to run the code.

In addition, the following libraries are also required. 
- pickle
- re
- sqlite3
- pandas
- flask
- joblib
- nltk
- numpy
- plotly
- sklearn
- sqlalchemy
- xgboost

## Project Motivation <a name="motivation"></a>

Through machine learning supervised model, emergency messages can be promptly categorized and sent to the appropriate disaster relief agency during disaster events when they are constrained by time and capacity. <br>
This project includes a web app enabling emergency workers to input a new message and get multi-output classification results in 36 pre-defined categories. In addition, the web app provides data visualzation on the genre and categories of those message data.

## Data <a name="data"></a>

The emergency data is provided by [Figure Eight](https://appen.com/). The data contains ~26000 messages of 3 genres (i.e. direct, social, news) and of 36 categories (e.g. aid related, medical help, seach and rescue).


## File Descriptions <a name="files"></a>

**Jupyter Notebooks:**

- `ETL Pipeline Preparation.ipynb`
- `ML Pipeline Preparation.ipynb`: try out various models and hyperparameters

**ETL and Machine Learning Pipeline:**
-  ETL pipeline: `data/process_data.py`
	- merge **mesages** and **categories** datasets
	- clean, transform, and save the data to SQLite database

- ML pipeline: `models/train_classifier.py`
	- load data from the SQLite database
	- build the model using sklearn's Pipeline with custom tokenize function and [feature union with heterogeneous data sources](https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html)
	- tune model using GridSearchCV
	- evaluate the model based on a combination of metrics (i.e. F1-minor, precision-minor, recall-minor, accuracy) on test data.
	- save the final model as a pickle file

**Flask Web App:**
- **app/run.py**: 
	- take user inputs
	- predict classification result based on user inputs
	- render flask app templates

- **app/master.html**:
	- structure home page layout
	- show two visualization on data using plotly

- **app/go.html**:
	- structure classifcation result page layout
	- show classification result visuals

## Instructions <a name="instructions"></a>

- **Run ETL pipeline:**`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv`
- **Run ML pipeline:**`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- **Run web app:**`python run.py` and go to `http://0.0.0.0:3001/`
