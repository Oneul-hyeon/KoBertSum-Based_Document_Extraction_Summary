import gradio as gr
from evaluate import get_summary

def ext_summarization(text) :
    result = get_summary(text, 3)
    return result

demo = gr.Interface(fn=ext_summarization, inputs = gr.Textbox(), outputs = gr.Textbox(), title = "추출 요약")
demo.launch(server_name = "0.0.0.0", server_port = 11021)