import tensorflow as tf
from flask import Flask, request, render_template
from bert4keras.tokenizer import Tokenizer
from gRPC import request_server

app = Flask(__name__)

maxlen = 300

dict_path = 'E:/Pretrained_Model/chinese_L-12_H-768_A-12/vocab.txt'

BERTtokenizer = Tokenizer(dict_path, do_lower_case=False)

pb_model_url = "0.0.0.0:8500"

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/tokenize/', methods=['POST', 'GET'])
def tokenizer():
    if request.method == 'POST':
        if request.form['text']:
            text = request.form['text']
            token_ids, segment_ids = BERTtokenizer.encode(text, max_length=maxlen)
            feature = {"token_ids": token_ids, "segment_ids": segment_ids}
            response = request_server(feature, pb_model_url)
            return response
    title = request.args.get('title', 'Default')
    return render_template('login.html', title=title)

@app.route('/classify/', methods=['POST', 'GET'])
def classify():
    pass


if __name__ == '__main__':
    app.run(debug=True)
