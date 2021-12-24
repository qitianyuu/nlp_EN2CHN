from flask import Flask, request, jsonify
from one_trans import machine_translate


app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route("/translation", methods=["post"])
def translate():
    # 翻译接口
    result = {}
    if request.method == 'POST':
        data = request.json
        # print(data, type(data))
        sentence = data.get('sentence')
        print(sentence)
        cn_trans = machine_translate(sentence)
        result['result'] = cn_trans
        result['msg'] = 'success'
        result['code'] = 200
    res = jsonify(result)
    return res




if __name__ == '__main__':
    app.run()
