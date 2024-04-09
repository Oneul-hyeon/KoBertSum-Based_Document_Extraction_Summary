from rouge import Rouge
import pandas as pd

result_df = pd.read_csv("result_df.csv")
# 예측된 요약문과 실제 요약문의 리스트
predicted_summaries = result_df["pred"]
actual_summaries = result_df["extractive_sents"]

rouge = Rouge()
rouge_score = rouge.get_scores(predicted_summaries, actual_summaries, avg = True)

# 평균 점수 출력
print("-----------------------------------------------\n")
print(f"Average ROUGE-1 Precision: {rouge_score['rouge-1']['p']:.2f}, Recall: {rouge_score['rouge-1']['r']:.2f}, F1: {rouge_score['rouge-1']['f']:.2f}")
print(f"Average ROUGE-2 Precision: {rouge_score['rouge-2']['p']:.2f}, Recall: {rouge_score['rouge-2']['r']:.2f}, F1: {rouge_score['rouge-2']['f']:.2f}")
print(f"Average ROUGE-L Precision: {rouge_score['rouge-l']['p']:.2f}, Recall: {rouge_score['rouge-l']['r']:.2f}, F1: {rouge_score['rouge-l']['f']:.2f}")
print("\n-----------------------------------------------")