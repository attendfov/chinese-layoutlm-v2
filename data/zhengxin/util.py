import traceback

import sys
import os
import uuid

from munch import DefaultMunch, Munch

# from conf.env_conf import PROJECT_DIR


def mdict(d={}):
     return DefaultMunch.fromDict(d)

def sortdict(dc):
    if type(dc) == dict:
        for k in dc:
            if type(dc[k]) == dict:
                dc[k]=sortdict(dc[k])
            if type(dc[k]) == list:
                dc[k] = sortdict(dc[k])
        return dict(sorted(dc.items(), key=lambda d: d[0], reverse=False))

    for i, subdict in enumerate(dc):
        if type(dc[i]) == dict:
            dc[i] = sortdict(dc[i])
    return dc

def findResource(name):
    for path in sys.path:
        if os.path.exists(os.path.join(path,name)):
           return os.path.join(path,name)

def get_uuid():
    return ''.join(str(uuid.uuid4()).split('-'))

def get_filedata(file,text=False,encoding='utf-8'):
    if text:
        with open(file, 'r',encoding=encoding) as f:
            return f.read()
    else:
        with open(file, 'rb') as f:
            return f.read()

def write_file(file,lines):
    with open(file,'w',encoding='utf-8') as f:
        if type(lines) is list:
            lines = [line.strip()+'\n' for line in lines]
            f.writelines(lines)
        else:
            f.write(lines)

def pdir(file:str):
    '''
        file: 相对工程的路径
        return : 绝对路径
    '''
    return f'{PROJECT_DIR}{file}' if not file.startswith('/') else file

def convert2rgb(img):
    '''
    在原图上进行bbox标记的时候需要先进行rgb转换不然灰度图或者二值图会报错
    @param img:
    @return:
    '''
    if (len(img.getbands()) != 3) or (img.format != 'JPEG') or (img.mode != 'RGB'):
        return img.convert('RGB')
    return img

def suppress_error(func):
    def inner(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except:
            traceback.print_exc()

    return inner

if __name__ == '__main__':
    print(findResource())