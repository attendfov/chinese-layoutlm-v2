from flask import request, Flask, send_file
import base64
import os
import numpy as np

app = Flask(__name__)

base_directory = os.path.dirname(os.path.abspath(__file__))

@app.route("/result/<filename>")
def download(filename):
    localfile = os.path.join(base_directory, 'requests_images', filename)
    if os.path.isfile(localfile):
        return send_file(localfile, as_attachment=True)
    else:
        return {'state': 'file is not existed'}


if __name__ == "__main__":
    app.run(host='localhost', port=10004, debug=True)