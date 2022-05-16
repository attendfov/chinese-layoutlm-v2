import os.path

import base64
import glob
from io import BytesIO

import pandas as pd
import time
import PIL
import requests
import json

from PIL import Image, ImageFont
from PIL import ImageDraw

# 接口文档地址：
# http://confluence.mycyclone.com/pages/viewpage.action?pageId=23335296#id-05_%E6%9C%8D%E5%8A%A1%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89v20211210-%E4%B8%80%E3%80%81%E9%80%9A%E7%94%A8OCR%E8%AF%86%E5%88%AB%E6%9C%8D%E5%8A%A1
# from conf.env_conf import PROJECT_DIR
from util import get_filedata, findResource, write_file, convert2rgb

def cyclone_ocr(target_img=''):
    host = f"http://172.25.128.118:6789"
    imgdata = base64.b64encode(get_filedata(target_img)).decode('utf-8')
    data = {
        "method": 0,
        'image': imgdata
    }
    res = requests.post(url=host, json=data)
    return json.loads(res.text)

def test_ocr(target_img=''):
    print(f'test_ocr start,target_img={target_img}')
    host = f"http://172.25.128.118:6789"
    imgdata = base64.b64encode(get_filedata(target_img)).decode('utf-8')
    data = {
        "method": 0,
        'image': imgdata
    }
    res = requests.post(url=host, json=data)
    res_obj = json.loads(res.text)

    write_file(f'{os.path.splitext(target_img)[0]}-ocr.json', json.dumps(res_obj, indent=2, ensure_ascii=False))
    # draw image
    img = Image.open(target_img)  # type:Image.Image
    img = convert2rgb(img)
    rimg_draw = ImageDraw.Draw(img)

    probs = []
    for obj in res_obj['objects']:
        coord = obj['points'][0] + obj['points'][2]
        probs.append(obj['detection_prob'])
        rimg_draw.rectangle(coord, fill=None, outline=PIL.ImageColor.getrgb('red'))
        font = ImageFont.truetype(findResource("ukai.ttc"), 15, encoding="unic")
        xy = obj['points'][0]
        xy[1] -= 10
        rimg_draw.text(xy, obj['text'], fill=PIL.ImageColor.getrgb('blue'), font=font)

    df = pd.DataFrame(data={
        'A':probs
    })
    # print(df.describe())

    img.save(f'{target_img}-ocr.png')
    # print(f'result={json.dumps(res_obj,indent=2,ensure_ascii=False)}')

def test_main():
    test_ocr(target_img=r'data\批2\sample\scale1\乔凤梅\乔凤梅_0.png')


def run_test():
    import time
    data_path = r'data\批2\sample'
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('png') and not file.endswith('ocr.png'):
                img_path = os.path.join(root, file)
                cur_time = time.time()
                test_ocr(img_path)
                print(img_path)
                print('用时：%d'%(time.time() - cur_time))
                print('\n')

if __name__ == '__main__':
    # test_main()

    run_test()