import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    #read csv file first, setting index, dropping columns with missing values, showing dimensions after that
    df = pd.read_csv(file_path, index_col="ID")
    df_cleaned = df.dropna(axis=1)
    print(f"Dimension of the {file_path} is:")
    print(df_cleaned.shape)
    

#Running within script
load_data("CKD.csv")