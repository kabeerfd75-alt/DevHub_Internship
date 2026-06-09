
import gradio as gr
from transformers import pipeline

classifier = pipeline("text-classification",
                      model="textattack/bert-base-uncased-ag-news")

def predict(text):
    return classifier(text)[0]

gr.Interface(fn=predict,
             inputs="text",
             outputs="json",
             title="AG News Topic Classifier").launch()
