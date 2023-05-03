from flair.data import Sentence
from flair.nn import Classifier
from flask import Flask, request, render_template
import torch
from train import SentimentalLSTM
from transformers import AutoTokenizer

app = Flask(__name__)
model = torch.load('/models/sentiment.pth')
h = model.init_hidden(1)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global h
    userinput = request.form['userinput']
    tokenized = tokenizer.encode(userinput, add_special_tokens=True)
    sig_out, h = model(torch.tensor([tokenized]), h)
    return render_template('index.html', predicted_value = f'{sig_out}')

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')
