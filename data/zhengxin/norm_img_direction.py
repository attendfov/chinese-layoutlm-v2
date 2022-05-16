import io

import PIL
import json

import base64
import numpy
import pandas as pd
import requests

from typing import Union

import datetime
import os
import re
from io import BytesIO

import fitz  # fitz就是pip install PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from fitz import Page

pdf2pic_zoom = 2
def pdf2pic(pdfPath,save_to=''):
    with fitz.open(pdfPath) as mu_doc:  # type:Page
        for pg in range(mu_doc.pageCount):
            page = mu_doc[pg]  # type:Page
            mat = fitz.Matrix(pdf2pic_zoom, pdf2pic_zoom)
            pix = page.getPixmap(matrix=mat, alpha=False)
            pix.save(save_to if save_to else f'{os.path.splitext(pdfPath)[0]}-{pg}.png')

            # stream = pix.pil_tobytes(format="png", optimize=True)
            # img = Image.open(BytesIO(stream))  # type:Image.Image
            # img.save(f'{os.path.splitext(pdfPath)[0]}-{pg}.png')


def norm_img_direction(image_file:Union[str,bytes],save_to:str='',save_non_rotate=True):
    """
        标准化图像方向
    @param image_file:原始图片路径或者bytes
    @param save_to:需要保存到哪个图片
    @return: rotate_type or ( bytes_of_img,rotate_type )
            rotate_type 可选值 [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
    """
    img = Image.open(image_file if type(image_file) is str else BytesIO(image_file))# type:Image.Image
    max_probs_img = [0,None,img]
    for r in [None,Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]:
        img1 = img.transpose(r) if r else img# type:Image.Image
        mean_probs = get_ocr_prob(img2bytes(img1))
        if mean_probs > max_probs_img[0]:
            max_probs_img = [mean_probs,r,img1]
    print(f'norm_img_direction,best rotate={max_probs_img[:2]}')
    if save_to:
        if save_non_rotate or max_probs_img[1]:
            max_probs_img[2].save(save_to)
        return max_probs_img[1]
    else:
        return img2bytes(max_probs_img[2]),max_probs_img[1]

def img2bytes(img:Image.Image):
    bytesio = io.BytesIO()
    img.save(bytesio, format='PNG')
    return bytesio.getvalue()

def get_ocr_prob(imgdata):
    host = f"http://172.25.128.118:6789"
    imgdata = base64.b64encode(imgdata).decode('utf-8')
    data = {
        "method": 0,
        'image': imgdata
    }
    res = requests.post(url=host, json=data)
    res_obj = json.loads(res.text)

    # draw debug image
    # img = Image.open(BytesIO(imgdata))  # type:Image.Image
    # rimg_draw = ImageDraw.Draw(img)

    probs = []
    for obj in res_obj['objects']:
        probs.append(obj['detection_prob'])
        # coord = obj['points'][0] + obj['points'][2]
        # rimg_draw.rectangle(coord, fill=None, outline=PIL.ImageColor.getrgb('red'))
        # font = ImageFont.truetype("ukai.ttc", 15, encoding="unic")
        # xy = obj['points'][0]
        # xy[1] -= 10
        # rimg_draw.text(xy, obj['text'], fill=PIL.ImageColor.getrgb('blue'), font=font)

    # img.save(f'{target_img}-ocr.png')

    return numpy.mean(probs)


def get_filedata(file):
    with open(file, 'rb') as f:
        return f.read()

def test_main():
    # usage case 1
    input_img = 'test/data/08754213001045170697-0.png'
    save_img = input_img.replace('.png', '-r-best.png')
    print(norm_img_direction(input_img, save_img))


    # usage case 2
    input_img = 'test/data/to-rotate-0.png'
    save_img = input_img.replace('.png', '-r-best.png')
    imgbytes,stats = norm_img_direction(get_filedata(input_img))
    with open(save_img,'wb') as f:
        f.write(imgbytes)
    print(stats)


if __name__ == '__main__':
    # norm_img_direction(r'data\批2\test\曹莹_0.png', r'data\批2\test\曹莹_0-new.png')

    data_path = r'data\批2\标注'
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('png'):
                img_path = os.path.join(root, file)
                dirs = img_path.split('\\')
                dirs[2] = '标注-方向矫正'
                save_img_path = '/'.join(dirs)
                if os.path.exists(save_img_path):
                    continue
                if not os.path.exists(os.path.dirname(save_img_path)):
                    os.makedirs(os.path.dirname(save_img_path))
                norm_img_direction(img_path, save_img_path)
