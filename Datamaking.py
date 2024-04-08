import pandas as pd
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

# train_drop, test_drop_data를 통해 label에 None 값이 존재하는 데이터 확인 가능
train_jsonl_data, train_drop_data = make_dataset(TRAIN_DIR + "법률_train_original/train_original.json", 
                                                 TRAIN_DIR + "사설_train_original/train_original.json", 
                                                 TRAIN_DIR + "신문기사_train_original/train_original.json")
test_jsonl_data, test_drop_data = make_dataset(VALIDATION_DIR + "법률_valid_original/valid_original.json", 
                                                VALIDATION_DIR + "사설_valid_original/valid_original.json", 
                                                VALIDATION_DIR + "신문기사_valid_original/valid_original.json")

# ext/data/raw 폴더에 jsonl 형식의 데이터 저장
with open(RAW_DIR + "train.jsonl" , encoding= "utf-8",mode="w") as file: 
	for line in train_jsonl_data : file.write(json.dumps(line, ensure_ascii=False) + '\n')
with open(RAW_DIR + "test.jsonl", encoding='utf-8', mode="w") as file: 
	for line in test_jsonl_data : file.write(json.dumps(line, ensure_ascii=False) + '\n')   