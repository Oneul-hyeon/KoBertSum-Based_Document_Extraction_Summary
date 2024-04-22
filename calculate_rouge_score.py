from korouge_score import rouge_scorer
import numpy as np
import pandas as pd

result_df = pd.read_csv("result_df.csv")
# 예측된 요약문과 실제 요약문의 리스트
predicted_summaries = result_df["pred"]
actual_summaries = result_df["extractive_sents"]

# ROUGE 스코어 계산을 위한 Scorer 초기화
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 각 요약문 쌍에 대한 ROUGE 점수 계산
scores = [scorer.score(actual, predicted) for predicted, actual in zip(predicted_summaries, actual_summaries)]

# 각 메트릭별 점수 계산을 위한 초기화
rouge1_precision, rouge1_recall, rouge1_f1 = [], [], []
rouge2_precision, rouge2_recall, rouge2_f1 = [], [], []
rougeL_precision, rougeL_recall, rougeL_f1 = [], [], []

# 점수 추가
for score in scores:
    rouge1_precision.append(score['rouge1'].precision)
    rouge1_recall.append(score['rouge1'].recall)
    rouge1_f1.append(score['rouge1'].fmeasure)

    rouge2_precision.append(score['rouge2'].precision)
    rouge2_recall.append(score['rouge2'].recall)
    rouge2_f1.append(score['rouge2'].fmeasure)

    rougeL_precision.append(score['rougeL'].precision)
    rougeL_recall.append(score['rougeL'].recall)
    rougeL_f1.append(score['rougeL'].fmeasure)

# 각 메트릭별 평균 점수 계산
avg_rouge1_precision = np.mean(rouge1_precision)
avg_rouge1_recall = np.mean(rouge1_recall)
avg_rouge1_f1 = np.mean(rouge1_f1)

avg_rouge2_precision = np.mean(rouge2_precision)
avg_rouge2_recall = np.mean(rouge2_recall)
avg_rouge2_f1 = np.mean(rouge2_f1)

avg_rougeL_precision = np.mean(rougeL_precision)
avg_rougeL_recall = np.mean(rougeL_recall)
avg_rougeL_f1 = np.mean(rougeL_f1)


# 평균 점수 출력
print("-----------------------------------------------\n")
print(f"Average ROUGE-1 Precision: {avg_rouge1_precision:.2f}, Recall: {avg_rouge1_recall:.2f}, F1: {avg_rouge1_f1:.2f}")
print(f"Average ROUGE-2 Precision: {avg_rouge2_precision:.2f}, Recall: {avg_rouge2_recall:.2f}, F1: {avg_rouge2_f1:.2f}")
print(f"Average ROUGE-L Precision: {avg_rougeL_precision:.2f}, Recall: {avg_rougeL_recall:.2f}, F1: {avg_rougeL_f1:.2f}")
print("\n-----------------------------------------------")