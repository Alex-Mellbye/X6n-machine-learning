
# In this document I prepare the dataset for machine learning analysis

import gc 
import re 
import pandas as pd
import numpy as np
import sklearn as skl 
import matplotlib.pyplot as plt 
import seaborn as sns 

df1=pd.read_sas(r"******, format = 'sas7bdat', encoding="latin-1")


#################### Now I start managing variables, recoding values, making dummies, and removing some missing data 

df1['Gender'] = df1['sex'].map({'Fema': 0, 'Male': 1})
df1 = df1.drop(["sex"], axis=1)


df1['Omsorgsniva'] = df1['omsorgsniva3'].map({1: "Inpatient", 2: "Day_treatment", 3: "Outpatient"})
df1 = df1.drop(["omsorgsniva3"], axis=1)


df1['Kontakttype'] = df1['kontaktType'].map({1: "Examination", 2: "Treatment", 3: "Controll"})
df1 = df1.drop(["kontaktType"], axis=1)
df1["Kontakttype"] = df1["Kontakttype"].fillna(0)


df1 = df1.drop(["innmateHast"], axis=1) #all values are 1


df1 = df1.drop(["lopenr", "dateinn", "dateutt"], axis=1) 


df1 = pd.get_dummies(df1, columns=["Omsorgsniva", "Kontakttype"])

df1 = df1.drop(["Kontakttype_0"], axis=1)      # Removing missing values
df1["index_var"] = df1.index                # Creating an identifier variable 

                                               

def f(row):                                                              # Here I am creating the variable dsh_yes out of the variable sh_ttgroup. For dsh_yes
    if (row['sh_ttgroup'] >= 1) & (row["sh_ttgroup"] <= 8):              # we only want those cases where sh_ttgroup equals 1 to 8
        val = 1
    else:
        val = 0
    return val


df1['dsh_yes'] = df1.apply(f, axis=1)



#################### With the initial dataset prepared I take the next step and start removing other variables that are not of interest 


df2 = df1


df2 = df2.drop(['bi_tilstand1', 'bi_tilstand2', 'bi_tilstand3', 'bi_tilstand4', 'bi_tilstand5', 
                  'bi_tilstand6', 'bi_tilstand7', 'bi_tilstand8', 'bi_tilstand9', 'bi_tilstand10',
                  'bi_tilstand11', 'bi_tilstand12', 'bi_tilstand13', 'bi_tilstand14', 'bi_tilstand15',
                  'bi_tilstand16', 'bi_tilstand17', 'bi_tilstand18', 'bi_tilstand19', 'bi_tilstand20',
                  'h_tilstand'], axis=1) 


df2["index_var2"] = df1["index_var"]



#################### Lastly I prepare the final dataset that is to be used for analysis. This consists of specifically selecting     
#################### the variables that make up sh_ttgroup (dsh_yes) and preparing df6 with the variable dsh_yes for later matching                            


df3 = df2 


df3 = df2[["X6n_yes", 
           "test_gr01", "test_gr02", "test_gr03", "test_gr04", "test_gr05", "test_gr06", "test_gr07", "test_gr08", 
           "psykbiyes", "Psyki", "pk_in2",
           "bi_sh_in", "sh_in", "sh_in2",
           "bi_sh1", "sh_poismed", "dsh1",
           "index_var"]]


# Here I am copying the identifier variable and putting it in df6 so that I can later match df6 on the results of the analysis

df6 = df2[["dsh_yes", "index_var2", "dsh2", "dsh3", "dsh4", "dsh5", "Gender", "age"]]






