import gradio as gr
from evaluate import get_summary
from test_info import get_info
from rouge import Rouge

def ext_summarization(text) :
    result = get_summary(text, 3)
    label = get_info(text)

    rouge = Rouge()
    rouge_score = rouge.get_scores(result, label, avg = True)

    return_rouge = f""
    for key in rouge_score :
        value = rouge_score[key]
        return_rouge += f"{key.upper()}\n"
        return_rouge += f"     > Precision : {value['p']:.2f}\n"
        return_rouge += f"     > Recall : {value['r']:.2f}\n"
        return_rouge += f"     > F-measure : {value['f']:.2f}\n"
    return result, label, return_rouge[:-2]

demo = gr.Interface(fn=ext_summarization,
                    inputs = gr.Textbox(),
                    outputs = [gr.Textbox(label = "추출 요약"), gr.Textbox(label = "정답 요약"), gr.Textbox(label = "Rouge Score")],
                    title = "추출 요약")

demo.launch(server_name = "0.0.0.0", server_port = 11021)