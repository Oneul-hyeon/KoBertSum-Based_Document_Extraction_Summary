import pandas as pd

def get_info(text) :
    try :
        return list(test_df[test_df["full text"] == text]["extractive_sents"])[0]
    except :
        return None

test_df = pd.read_csv("test_df.csv")