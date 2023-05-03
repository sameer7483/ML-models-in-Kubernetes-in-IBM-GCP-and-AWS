from flask import Flask, request, render_template
import torch
from torch import nn
from torchtext.datasets import AG_NEWS
train_iter = iter(AG_NEWS(split='train'))

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))

app = Flask(__name__)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

train_iter = AG_NEWS(split='train')
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

model.load_state_dict(torch.load('../train/ngram.pt'))
# model = torch.load('../train/ngram.pth')
model.eval()

def predict_news(text, text_pipeline):
    print(text)
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}
    userinput = request.form['userinput']
    print(userinput, 'ritikaaa')
    return render_template('index.html', predicted_value = f'{ag_news_label[predict_news(userinput, text_pipeline)]}')
    # tokenized = tokenizer.encode(userinput, add_special_tokens=True)
    # sig_out, h = model(torch.tensor([tokenized]), h)
    # pred = torch.round(sig_out.squeeze()) 
    # return render_template('index.html', predicted_value = f'{pred}')

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')
