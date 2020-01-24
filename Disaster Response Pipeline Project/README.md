# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)
5. [Instructions](#instructions)

## Installation <a name="installation"></a>

This code runs with Python version 3.* To run the code here, you'll need along with the Anaconda distribution of Python the following packages need to be installed:
* punkt
* wordnet
* averaged_perceptron_tagger


## Project Motivation<a name="motivation"></a>

This is a project requirement for the DSND. The scope of this project is to implement the Data Engineering skills through building pipelines, mainly ETL and ML pipelines.
In simple terms, the goal is to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. It includes a web app where an emergency worker can input a new message and get classification results in several categories. 


## File Descriptions <a name="files"></a>

There are three main foleders:
1. app
    - templates: contain html files for the web app
    - run.py: it is a Flask file to run the web app

2. data
    - DisasterResponse.db: a database output of the ETL pipeline which contains the messages and categories data
    - ETL Pipeline Preparation.ipynb: a notebook that contain all the processing before deploying it to the .py file
    - disaster_categories.csv: dataset including all the categories 
    - disaster_messages.csv: dataset including all the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    
3. models
    - ML Pipeline Preparation.ipynb: a notebook that contain all the processing before deploying it to the .py file
    - classifier.pkl: a classifier trained as an output of the machine learning pipeline
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits must be given to Udacity for the codes outline and FigureEight for provding the data used by this project. 

## Instructions:<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
