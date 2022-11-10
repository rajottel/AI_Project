from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sb

df_training = pd.read_csv ("Data/aps_failure_training_set.csv")
df_test = pd.read_csv ("Data/aps_failure_test_set.csv")

def get_correct_label(df_training):
    return df_training.replace(['neg','pos'],[0,1])

print(df_training)