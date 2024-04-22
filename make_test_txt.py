import os
import pandas as pd

try : os.mkdir("test_txt")
except : pass

DIR = "test_txt"
data = pd.read_csv("test_df.csv")
for index, info in data.iterrows() :
    id, full_text = info["id"], info["full text"]
    f = open(f"{DIR}/{id}.txt", "+w")
    f.write(full_text)
    f.close()