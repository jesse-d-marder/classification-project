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
| customer_id    |  unique id identifying customer          |
| is_senior_citizen           |  1 if the customer is a senior citizen           |
| tenure    |  number of months customer has been with Telco      |
| multiple_lines           |  whether the customer has multiple phone lines - Yes, No, or No Phone Service|
| online_security    |  whether the customer has online security option         |
| online_backup    |  whether the customer has online backup option         |
| device_protection   |  whether the customer has device protection option         |
| tech_support    |  whether the customer has tech support option         |
| streaming_tv    |  whether the customer has streaming tv option         |
| streaming_movies    |  whether the customer has streaming movies option         |
| monthly_charges    |  dollar amount of customer monthly charge         |
| total_charges    |  dollar amount of total amount charged to customer         |
| churn    |  whether the customer churned or not, the target        |
| contract_type    |  whether the customer has month-to-month, 1 year, or 2 year contract       |
| internet_service_type    |  whether the customer has Fiber optic, DSL, and No internet        |
| payment_type    |  the method customer uses to pay (manual check, electronic check, EFT, credit card)        |
| is_male    |  whether the customer is male      |
| has_phone    |  whether the customer has a phone       |
| has_internet_service    |  whether the customer has internet or not        |
| has_partner    |  whether the customer has a partner or not       |
| has_dependent    |  whether the customer has a dependent or not       |
| is_paperless    |  whether the customer has paperless bulling      |
| is_month_to_month    |  whether the customer has a month to month contract or long term         |
| is_autopay    |  whether the customer pays through automatic means or manual     |
| has_streaming    |  whether the customer has streaming        |




### Project Plan

For this project I followed the data science pipeline:

Planning: I established the goals for this project and the relevant questions I want to answer. I developed a Trello board (https://trello.com/b/yCRSVyiw/telco-churn-classification-project) to help keep track of open and completed work.

Acquire: The data for this project is from a SQL Database called 'telco_churn'. The acquire.py script is used to query the database for the required data tables and returns the data in a Pandas DataFrame. This script also saves the DataFrame to a .csv file for faster subsequent loads. The script will check if the telco_churn.csv file exists in the current directory and if so will load it into memory, skipping the SQL query.

Prepare: The prepare.py script has a prep_telco function that takes as an argument the Pandas DataFrame acquired from acquire.py and prepares the data for exploration and modeling. Steps here include removing null values (NaN), converting data types, and encoding categorical variables. The function also drops unnecessary columns. This script also contains a train_validate_test_split function to split the dataset into train, validate, and test sets cleanly.

Explore: The questions established in planning were analyzed using statistical tests including chi-squared and t-test to test hypotheses about the data. This work was completed in the explore.ipynb file and relevant portions were moved to the churn_report.ipynb final deliverable. A visualization illustrating the results of the tests and answering the question is included. 

Model: Four different classification algorithms were investigated to determine if churn could be predicted using features identified during exploration. A select set of hyperparameters were tested against train and validate data to determine which demonstrated the best performance. The final model was selected and used to make predictions on the withheld test data. These predictions are included in the predictions.csv in this repository.

Delivery: This is in the form of this github repository as well as a presentation of my final notebook to the stakeholders.

### Steps to Reproduce

1. You will need an env.pu file that contains the hostname, username and password of the mySQL database that contains the telco_churn table. Store that env file locally in the repository. 
2. Clone my repository (including the acquire.py and prepare.py). Confirm .gitignore is hiding your env.py file/
3. Libaries used are pandas, matplotlib, scipy, sklearn, seaborn, and numpy.
4. You should be able to run churn_report.ipynb.

### Key Findings and Takeaways

- A few groups churn at rates greater than the average. These include:
    - month to month plan customers
    - those with internet service
- Looking closer at the month to month plan partipants, those who churn do pay more on average than those who stay. This implies cost may be a driver of churn for these customers.
- Examining the internet service customers, fiber customers do churn on average 10% more than we'd expect. Perhaps our fiber service needs to be enhanced.
- Testing whether having streaming or not for our fiber customers affects whether they will churn those with streaming churn at a lower rate than those who don't have streaming. Still - churn rates for both streamers and non-streamers are above the overall average.
- Of the four classification algorithms evaluated to predict churn, each with a variety of hyperparameter settings for a total of 311 models, k-nearest neighbors with a k value of 87 demonstrated the best performance overall. Tested this model on the test set resulted in an accuracy of 79% - an improvement of 6% from the baseline.