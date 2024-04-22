import gradio as gr
from evaluate import get_summary
from test_info import get_info
from korouge_score import rouge_scorer

def ext_summarization(text) :
    result = get_summary(text, 3, "KoBigBird")
    label = get_info(text)
    if label != None : 
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        rouge_score = scorer.score(label, result)
        return_rouge = f""
        for key in rouge_score.keys() :
            value = rouge_score[key]
            return_rouge += f"{key.upper()}\n"
            return_rouge += f"     > Precision : {value.precision:.2f}\n"
            return_rouge += f"     > Recall : {value.recall:.2f}\n"
            return_rouge += f"     > F-measure : {value.fmeasure:.2f}\n"
        return result, label, return_rouge[:-2]
    else : return result, "None", "None"
    
demo = gr.Interface(fn=ext_summarization,
                    inputs = gr.Textbox(),
                    outputs = [gr.Textbox(label = "예측 요약"), gr.Textbox(label = "정답 요약"), gr.Textbox(label = "Rouge Score")],
                    title = "KoBigBird 기반 추출 요약")

demo.launch(server_name = "0.0.0.0", server_port = 11021)