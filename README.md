## This repository contains the code for the classification project completed as part of the Codeup Data Science curriculum. 

## Repo contents:
### 1. This Readme:
    - Project description with goals
    - Inital hypothesis/questions on data, ideas
    - Data dictionary
    - Project planning
    - Instructions for reproducing this project and findings
    - Key findings, recommendations and takeaways from project
### 2. Final report (Jupyter Notebook)
### 3. Acquire module
### 4. Prepare module
### 5. Predictions .csv
### 6. Exploration & modeling notebook

### Project Goals

The goal of this project is to identify drivers of customer churn at Telco. I will try to answer the question of why customers churn and provide recommendations for reducing customer churn.

### Initial Questions and Hypotheses

1. Do any groups of customers churn at higher than the overall average rate?
2. Do any groups of customers churn at lower than the overall average rate?
3. For the month to month plan - do churners pay more than non churners? 
4. Of the groups that churn at higher than overall average, do any particular sub groups churn higher?
5. Within the sub groups do any specific options lead to higher churn?

### Data Dictionary

| Variable    | Meaning     |
| ----------- | ----------- |
| has_streaming    |  whether the customer has either streaming TV or Movies or both           |
|             |             |

### Project Plan

For this project I followed the data science pipeline:

Planning: I established the goals for this project and the relevant questions I want to answer. I developed a Trello board (https://trello.com/b/yCRSVyiw/telco-churn-classification-project) to help keep track of open and completed work.

Acquire: The data for this project is from a SQL Database called 'telco_churn'. The acquire.py script is used to query the database for the required data tables and returns the data in a Pandas DataFrame. This script also saves the DataFrame to a .csv file for faster subsequent loads. The script will check if the telco_churn.csv file exists in the current directory and if so will load it into memory, skipping the SQL query.

Prepare: The prepare.py script has a prep_telco function that takes as an argument the Pandas DataFrame acquired from acquire.py and prepares the data for exploration and modeling. Steps here include removing null values (NaN), converting data types, and encoding categorical variables. The function also drops unnecessary columns. This script also contains a train_validate_test_split function to split the dataset into train, validate, and test sets cleanly.

Explore: 

Model:

Delivery: This is in the form of this github repository as well as a presentation of my final notebook to the stakeholders.

### Steps to Reproduce

1. You will need an env.pu file that contains the hostname, username and password of the mySQL database that contains the telco_churn table. Store that env file locally in the repository. 
2. Clone my repository (including the acquire.py and prepare.py). Confirm .gitignore is hiding your env.py file/
3. Libaries used are pandas, matplotlib, scipy, sklearn, seaborn, and numpy.
4. You should be able to run churn_report.ipynb.

### Key Findings and Takeaways