import torch
from main import text_pipeline
from flask import Flask, request, render_template
from main import TextClassificationModel
app = Flask(__name__)
model = torch.load('./ngram.pth')
model.eval()
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    userinput = request.form['userinput']
    with torch.no_grad():
        text = torch.tensor(text_pipeline(userinput))
        output = model(text, torch.tensor([0]))
        prediction = ag_news_label[output.argmax(1).item() + 1]
    return render_template('index.html', predicted_value = f'{prediction}')

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')
