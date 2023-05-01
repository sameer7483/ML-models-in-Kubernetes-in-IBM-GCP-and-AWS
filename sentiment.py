from flair.data import Sentence
from flair.nn import Classifier
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # make a sentence
    userinput = request.form['userinput']
    sentence = Sentence(userinput)

    # load the NER tagger
    tagger = Classifier.load('sentiment')

    # run NER over sentence
    tagger.predict(sentence)

    # print the sentence with all annotations
    print(sentence)
    return render_template('index.html', predicted_value = f'{sentence}')

if __name__ == '__main__':
    app.run(debug=True)

