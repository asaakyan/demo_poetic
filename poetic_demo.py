from flask import *
import os
# import torch
import os
import time
import numpy as np
import sys
# from search_and_select_evidence_for_claim import *

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
	if (request.method == "POST"):
		poem = request.form["poem"]
		print('poem:',  poem)
		return render_template("translation.html", poem=poem, translation=get_transaltion(poem))

	return render_template("index.html", f='')

def get_transaltion(poem):
	translation = "Lorem ipsum"
	return translation 

# @app.route("/returnev", methods=["GET", "POST"])
# def returnev():
# 	if (request.method == "POST"):
# 		evlist = [e.split('\r\n')[0] for e in request.form.getlist('mycheckbox')]
# 		print(evlist)
# 		with open('claim.txt', 'r') as outf:
# 			claim = outf.read()

# 		if not evlist or "no_evid" in evlist:
# 			veracity = "No evidence"
# 			return render_template("index3.html", veracity=veracity, claim=claim, score='N/A',\
# 				ev6="N/A", ev5="N/A",ev4="N/A", ev3="N/A", ev2="N/A", ev1="N/A")

# 		tokens = roberta.encode(" ".join(evlist), claim.capitalize())
# 		prediction = roberta.predict('sentence_classification_head', tokens)
# 		probabilities = softmax(prediction.cpu().detach().numpy(), axis=1)

		
# 		proba = probabilities[0][1]
# 		if proba < 0.8:
# 			veracity = 'REFUTED'
# 			col="red"
# 			proba = 1-proba #confidence how much refuted
# 		elif proba >= 0.8:
# 			veracity = 'SUPPORTED'
# 			col="green"
# 			# proba stays the same: confidence how much supported
# 		else:
# 			veracity = 'ERROR'
		
# 		if len(evlist) == 6:
# 			return render_template("index3.html", claim=claim,
# 				veracity=veracity, score="{:.2f}".format(proba), col=col, ev1=evlist[0], ev2=evlist[1], ev3=evlist[2], ev4=evlist[3], ev5=evlist[4], ev6=evlist[5] )
# 		elif len(evlist) == 5:
# 			return render_template("index3.html",  claim=claim,
# 				veracity=veracity, score="{:.2f}".format(proba), col=col, ev1=evlist[0], ev2=evlist[1], ev3=evlist[2], ev4=evlist[3], ev5=evlist[4], ev6="N/A")
# 		elif len(evlist) == 4:
# 			return render_template("index3.html",  claim=claim,
# 				veracity=veracity, score="{:.2f}".format(proba), col=col, ev1=evlist[0], ev2=evlist[1], ev3=evlist[2], ev4=evlist[3], ev6="N/A", ev5="N/A")
# 		elif len(evlist) == 3:
# 			return render_template("index3.html",  claim=claim,
# 				veracity=veracity, score="{:.2f}".format(proba), col=col, ev1=evlist[0], ev2=evlist[1],ev3=evlist[2], ev6="N/A", ev5="N/A", ev4="N/A")
# 		elif len(evlist) == 2:
# 			return render_template("index3.html",  claim=claim,
# 				veracity=veracity, score="{:.2f}".format(proba), col=col, ev1=evlist[0], ev2=evlist[1], ev6="N/A", ev5="N/A", ev4="N/A", ev3="N/A")
# 		else:
# 			return render_template("index3.html",  claim=claim,
# 				veracity=veracity, score="{:.2f}".format(proba), col=col, ev1=evlist[0], ev6="N/A", ev5="N/A",ev4="N/A", ev3="N/A", ev2="N/A")


if __name__ == '__main__':

	np.random.seed(42)
	# torch.manual_seed(42)

	# roberta = RobertaModel.from_pretrained(
	# 	'./covidfact-roberta/',
	# 	checkpoint_file='checkpoint_best.pt',
	# 	data_name_or_path='./RTE-covidfact-bin'
	# )
	# roberta.cuda()
	# roberta.eval()
	# print("RoBERTa loaded in main")

	# sbert = SentenceTransformer('paraphrase-TinyBERT-L6-v2', device='cuda')
	# print("SBERT loaded in main")

	app.run(host='0.0.0.0')
