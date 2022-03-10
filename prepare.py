import pandas as pd
import numpy as np
from env import get_db_url
from sklearn.model_selection import train_test_split

def prep_telco(df):
    # Drop unnecessary foreign key ids
    df = df.drop(columns=['payment_type_id','internet_service_type_id','contract_type_id'])
    # From prior exploration of dataset a small number of the total_charges are just whitespace - these are all new customers who haven't been with the company for >1 month.
    # Given that it is a very small proportion of the total dataset these rows will be deleted for ease of computation later on
    df = df.drop(df[df.total_charges == " "].index)
    # Convert total_charges to float for later analysis
    df.total_charges = df.total_charges.astype('float64')

    # Determine the categorical variables - here defined as object data type (non-numeric) and with fewer than 5 values
    catcol = df.columns[(df.nunique()<5)&(df.dtypes == 'object')]
    # Encode categoricals
    dummy_df = pd.get_dummies(df[catcol], dummy_na=False, drop_first=True)
    # Concatenate dummy df to original df
    df = pd.concat([df,dummy_df],axis=1)
    # Remove the original categorical columns after encoding
    df = df.drop(columns=catcol)
    
    return df

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

