from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    #read csv file first, setting index, dropping columns with missing values, showing dimensions after that
    df = pd.read_csv(file_path, index_col="ID")
    df_cleaned = df.dropna(axis=1)
    print(f"Dimension of the {file_path} is:")
    print(df_cleaned.shape)
    
    #encode the label
    df_cleaned["label"] = df_cleaned["label"].map({"CKD": 1, "Normal": 0})
    
    #df_cleaned = df_cleaned.drop(columns=["label"])
    print("First 5 samples:")
    print(df_cleaned.head())
    
    #convert df to numpy
    np_array = df_cleaned.to_numpy()
    
    #split the features and the label
    X = np_array[:,:-1]
    y = np_array[:,-1]
    return X,y


    
    

#Running within script
load_data("CKD.csv")