import pandas as pd
import numpy as np
import json

def make_dataset(law_json, editorial_json, news_json) :
    law_data = pd.read_json(law_json)["documents"]
    editorial_data = pd.read_json(editorial_json)["documents"]
    news_data = pd.read_json(news_json)["documents"]

    jsonl_data, drop_data = [], []
    for data in [law_data, editorial_data, news_data] :
        for row in data :
            sequence = []
            for text_set in row["text"] :
                for text in text_set :
                    sequence.append(text["sentence"])
            if None not in row['extractive'] :
                jsonl_data.append({'id' : row['id'], 'article_original' : sequence, 'extractive' : row['extractive']})
            else : 
                drop_data.append({'id' : row['id'], 'article_original' : sequence, 'extractive' : row['extractive']})
    return jsonl_data, drop_data

TRAIN_DIR = "json_raw_data/Training/"
VALIDATION_DIR = "json_raw_data/Validation/"
RAW_DIR = 'ext/data/raw/'

test_jsonl_data, test_drop_data = make_dataset(VALIDATION_DIR + "법률_valid_original/valid_original.json", 
                                                VALIDATION_DIR + "사설_valid_original/valid_original.json", 
                                                VALIDATION_DIR + "신문기사_valid_original/valid_original.json")
df = pd.DataFrame(test_jsonl_data)
df["full text"] = df.article_original.apply(lambda x : ' '.join(x))
df['extractive_sents'] = df.apply(lambda row: ' '.join(list(np.array(row['article_original'])[row['extractive']])) , axis=1)
df.to_csv("test_df.csv", index = False)