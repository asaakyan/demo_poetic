from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration, MBart50TokenizerFast
import os, sys
import numpy as np
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app) 

@app.route("/", methods=["GET", "POST"])
def home():
	if (request.method == "POST"):
		form = request.form
		poem = form["poem"]
		lang = form["lang"]
		model_name = form["model"]
		# print("lang:")
		# print(lang)
		# print("model:")
		# print(model)
		# print('poem:')
		# print(poem)
		poem_lines = poem.split('\n')
		return render_template("translation.html", poem=poem_lines, translation=get_transaltion(poem_lines, lang, model_name))

	return render_template("index.html", f='')

def get_transaltion(poem_lines, lang, model_name):
	trans_lines = []
	for line in poem_lines:
		trans_lines.append(translate(line, lang, model_name))
	return trans_lines 

def translate(src, src_lang, model_name):

	if model_name=='mbart-50':
		model = mbart50_model
		tokenizer = mbart50_tokenizer
	elif src_lang[:3]+model_name=='ru_poetic_all':
		model = ru_poetic_all_model
		tokenizer = ru_poetic_all_tokenizer
	else: return src.strip()[::-1]

	tokenizer.src_lang = src_lang
	encoded = tokenizer(src, return_tensors="pt")
	generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
	return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

if __name__ == '__main__':

	# np.random.seed(42)
	# torch.manual_seed(42)
	# os.environ["CUDA_VISIBLE_DEVICES"]="2"

	# MODEL_PATH = '/'
	# poetic_all_model = False #MBartForConditionalGeneration.from_pretrained(MODEL_PATH)
	# poetic_all_tokenizer = False #MBart50TokenizerFast.from_pretrained(MODEL_PATH)
	print("Loading poetic model")
	ru_poetic_all_tokenizer = AutoTokenizer.from_pretrained("TuhinColumbia/russianpoetrymany")
	ru_poetic_all_model = AutoModelForSeq2SeqLM.from_pretrained("TuhinColumbia/russianpoetrymany")

	print("Loading mBART model")
	#mbart50_model =  MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
	#mbart50_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

	#model_dict = {'ru_poetic-all': ru_poetic_all_model, 'mbart-50': mbart50_model}
	# tokenizer_dict = {'ru_poetic-all':ru_poetic_all_tokenizer, 'mbart-50': mbart50_tokenizer}

	app.run()
