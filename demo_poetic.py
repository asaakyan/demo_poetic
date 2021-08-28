from flask import Flask, request, render_template
# import torch
import os, sys
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
	if (request.method == "POST"):
		poem = request.form["poem"]
		print('poem:')
		print(poem)
		poem_lines = poem.split('\n')
		return render_template("translation.html", poem=poem_lines, translation=get_transaltion(poem_lines))

	return render_template("index.html", f='')

def get_transaltion(poem_lines):
	trans_lines = []
	for line in poem_lines:
		trans_lines.append(line.strip()[::-1])
	return trans_lines 

if __name__ == '__main__':

	np.random.seed(42)
	# torch.manual_seed(42)

	app.run(host='0.0.0.0')
