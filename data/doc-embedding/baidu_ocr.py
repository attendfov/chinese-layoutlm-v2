# encoding:utf-8

import requests
import base64

def get_access_token():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    ak = 'BMAv0vyjbUSqgtqgfZa35bmp'
    sk = 'BXXykWljc0QXrDyvbvTjqil04WVVTCib'
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s' % (ak, sk)
    response = requests.get(host)
    if response:
        print(response.json())

def general_ocr(img_path):
    '''
    通用文字识别
    '''

    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
    # 二进制方式打开图片文件
    f = open(img_path, 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img}
    access_token = '24.06c36877466f9e636fef9d75d83da8b0.2592000.1648275117.282335-16904434'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        print(response.json())





if __name__ == '__main__':
    # get_access_token()

    general_ocr('sample-data\zh_val_37.jpg')