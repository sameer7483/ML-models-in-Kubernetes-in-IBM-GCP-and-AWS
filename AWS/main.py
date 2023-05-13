import os
import torch
from flask import Flask, request, render_template, json, send_from_directory

from lstm import SentimentalLSTM


app = Flask(__name__)

train_on_gpu = False
port = 8000
os.environ["MODEL_DIR"] = "./model"

def get_model():

	device = 'cpu'
	model_dir = os.environ.get('MODEL_DIR')

	model_path = model_dir + '/' + 'sentiment.pth'
	# model_path = './model/sentiment_sd.pth'


	params = {
		vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
		output_size : 1,
		embedding_dim : 400,
		hidden_dim : 256,
		n_layers : 2
	}

	model = SentimentalLSTM().to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()

	print(model)
	return model

	# with torch.no_grad():
	# 	output = model(tensor)
	# 	_, predicted_class = torch.max(output, 1)

	# prediction = predicted_class.item()
	# return predicted_class.item()



def test(inputs):
	# from train code
	batch_size=50

	# load the model
	model = get_model()

	# run inference
	model.eval()

	# init hidden state
	h = model.init_hidden(batch_size)

	# Creating new variables for the hidden state, otherwise
	# we'd backprop through the entire training history
	h = tuple([each.data for each in h])

	output, h = model(inputs, h)

	# convert output probabilities to predicted class (0 or 1)
	pred = torch.round(output.squeeze())  # rounds to the nearest integer
	return pred


@app.route('/')
def hello_world():
     return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

	# make a sentence
	userinput = request.form["userinput"]
	print(userinput)

	# make the prediction
	pred = test(userinput)

	# print the sentence with all annotations
	print(pred)
	return render_template("index.html", predicted_value=f"{pred}")


app.run(port=port, debug=True)
