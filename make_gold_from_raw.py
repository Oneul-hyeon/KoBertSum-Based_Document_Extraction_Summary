import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_file_name", type=str, default="test_v2.jsonl")
args = parser.parse_args()
test_file_path = f"./ext/data/raw/{args.test_file_name}"
# gold_file_path = f"./ext/results/{args.test_file_name.split('.')[0]}.gold"
gold_file_path = f"./ext/results/test_v3.gold"
with open(test_file_path, "r") as file:
    with open(gold_file_path, "w") as gold_out:
        for jsline in file:
            data = json.loads(jsline)
            label = data['extractive']
            article = data['article_original']
            for ind in label:
                gold_sent = article[ind]
                gold_out.write(gold_sent + ' ')
            gold_out.write("\n")