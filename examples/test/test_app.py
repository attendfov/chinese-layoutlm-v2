# -*- coding: utf-8 -*-
import urllib

import os
import json
import traceback

import numpy as np
from flask import Flask, send_file

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

base_directory = os.path.dirname(os.path.abspath(__file__))

@app.route("/result/<filename>")
def download(filename):
    # filename = urllib.unquote(filename)
    localfile = os.path.join(base_directory,'result',filename)
    if os.path.isfile(localfile):
        return send_file(localfile, as_attachment=True)
    else:
        return None


if __name__ == "__main__":
    app.run(host="localhost",port=10003, debug=True)

