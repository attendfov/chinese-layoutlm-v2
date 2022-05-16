import os
import requests
import pdfplumber
import numpy as np
import json
import uuid
import shutil
import pandas as pd
import cv2
import fitz
from lxml import etree

from pdf2text_pdfminer_extract_imgs import extract_imgs
from pdf2pic import pyMuPDF_fitz
from ruizhen_ocr import RuizhenAngle

data = {"lang": "zh",
        "version": "0.1",
        "split": "train/val",
        "documents": [
            {
                'id': '',
                'uid': '',
                'label': 0,
                'document': [
                    {'id': '',
                     'linking': [],
                     'box': [],
                     'text': '',
                     'label': '',
                     'words': [
                         {'box': [],
                          'text': ''}
                     ]}
                ],
                "img": {
                    "fname": '',
                    "width": 0,
                    "height": 0
                }
            },

        ], }

documents = {"lang": "zh",
             "version": "0.1",
             "split": "train/val",
             'documents': []}

enlarge = 4
doc_id = 0
line_id = 0

with open('data/id2label.json', 'r', encoding='utf-8') as f:
    id2label = json.load(f)
    label2id = {v: k for k, v in id2label.items()}

datasets_labels = set()

ruizhen_ocr = RuizhenAngle()


def ocr_server(img_path):
    url = 'http://localhost:8001/ocr_rec'

    root = os.path.dirname(os.path.abspath(__file__))
    data = {'files': [os.path.join(root, img_path)]}

    response = requests.post(url, data=json.dumps(data))
    res = json.loads(response.content)
    return res


def dec_label(path):
    dirname = os.path.dirname(path).split('\\')
    dirname.remove('data')
    filename = os.path.basename(path)
    basename = os.path.splitext(filename)[0]
    if 'PO' in dirname:
        label = '-'.join(dirname)
    else:
        chars = list(basename)
        while chars[-1] in [str(i) for i in range(10)]:
            del chars[-1]
        basename_new = ''.join(chars)
        label = '-'.join(dirname) + '-' + basename_new
    # labels.add(label)
    return label2id[label]


def get_uuid():
    return ''.join(str(uuid.uuid4()).split('-'))


def rec_img_from_doc(img_save_dir, pdf_width, pdf_height, img_paths):
    '''
    判断提取的图片占盘面的比例，高于一定比例，直接对pdf转成的图片识别，否则
    提取的图片和文字解析共同执行
    :param img_save_dir: pdf中提取的图片
    :param pdf_width:
    :param pdf_height:
    :param img_paths: pdf转成的图片
    :return:
    '''
    global line_id, documents
    line_list = []

    # 读取单页pdf中的所有图片并识别
    for img_path in os.listdir(img_save_dir):
        # page_num + '$$' + x0 + '##' + y0 + '##' + x1 + '##' + y1 + '##' + width + '-' + height + '@@' + ext
        if img_path.endswith('jpg') or img_path.endswith('png'):
            basename = os.path.splitext(img_path)[0]
            page_num_img, partial_name = basename.split('$$')
            # x0是图片在原图中的位置
            x0, y0, x1, y1, name = partial_name.split('##')
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)

            if page_num_img != '0':
                continue

            # 图片的原始大小
            img_w, img_h = name.split('@@')[0].split('-')

            # y0,y1是以左下点为远点的y轴坐标，需要变换成左上角
            # 注意变换方法
            new_x0 = x0
            new_y0 = pdf_height - y1

            w_multiple = (float(x1) - float(x0)) / float(img_w)
            h_multiple = (float(y1) - float(y0)) / float(img_h)

            # 判断图片占据pdf版面的比例
            if (y1 - y0) * (x1 - x0) / (pdf_width * pdf_height) * 100 > 10:
                # 读取pdf转成的图片
                ocr_res = ruizhen_ocr.request_in_image_file(img_paths[0])
                rec_results = ocr_res['response']['data']['identify_results'][0]['details']['text2']

                for line in rec_results:
                    line_json = {'id': '',
                                 'linking': [],
                                 'box': [],
                                 'text': '',
                                 'label': '',
                                 'words': []}
                    line_box = [line['regions'][0], line['regions'][1],
                                line['regions'][2], line['regions'][3]]
                    line_box = [int(p) for p in line_box]
                    line_json['box'] = line_box
                    line_json['text'] = line['result']
                    word_list = []
                    for word in line['words']:
                        word_json = {'box': [], 'text': ''}
                        word_json['text'] = word['result']
                        word_box = [word['region'][0], word['region'][1],
                                    word['region'][2], word['region'][3]]
                        word_box = [int(p) for p in word_box]
                        word_json['box'] = word_box
                        word_list.append(word_json)
                    line_json['id'] = line_id
                    line_json['words'] = word_list
                    line_id += 1
                    line_list.append(line_json)
                break  # 只执行一次
            else:
                ocr_res = ruizhen_ocr.request_in_image_file(os.path.join(img_save_dir, img_path))
                rec_results = ocr_res['response']['data']['identify_results'][0]['details']['text2']

                for line in rec_results:
                    line_json = {'id': '',
                                 'linking': [],
                                 'box': [],
                                 'text': '',
                                 'label': '',
                                 'words': []}
                    line_box = [line['regions'][0] * w_multiple + new_x0, line['regions'][1] * h_multiple + new_y0,
                                line['regions'][2] * w_multiple + new_x0, line['regions'][3] * h_multiple + new_y0]
                    line_box = [int(p * enlarge) for p in line_box]
                    line_box = validate_bbox(line_box, img_paths)
                    line_json['box'] = line_box
                    line_json['text'] = line['result']
                    word_list = []
                    for word in line['words']:
                        word_json = {'box': [], 'text': ''}
                        word_json['text'] = word['result']
                        word_box = [word['region'][0] * w_multiple + new_x0, word['region'][1] * h_multiple + new_y0,
                                    word['region'][2] * w_multiple + new_x0, word['region'][3] * h_multiple + new_y0]
                        word_box = [int(p * enlarge) for p in word_box]
                        word_box = validate_bbox(word_box, img_paths)
                        word_json['box'] = word_box
                        word_list.append(word_json)
                    line_json['id'] = line_id
                    line_json['words'] = word_list
                    line_id += 1
                    line_list.append(line_json)

    return line_list


def validate_bbox(line_bbox, img_paths):
    img = cv2.imread(img_paths[0])
    h, w, c = img.shape
    if line_bbox[0] > w:
        line_bbox[0] = w
    if line_bbox[2] > w:
        line_bbox[2] = w
    if line_bbox[1] > h:
        line_bbox[2] = h
    if line_bbox[3] > h:
        line_bbox[3] = h
    return line_bbox


def doc_recognition(path, img_paths):
    global doc_id, line_id, documents

    img_save_dir = os.path.join('imgs_from_pdf', get_uuid())
    if os.path.exists(img_save_dir):
        shutil.rmtree(img_save_dir)
    os.makedirs(img_save_dir)
    # 一次性提取出所有图片
    extract_imgs(path, img_save_dir)

    # 获取文档标签
    label = dec_label(path)
    datasets_labels.add(label)

    doc = fitz.open(path)

    for pag_num, page in enumerate(doc):
        # 只考虑第一页
        if pag_num > 0:
            continue

        root = etree.fromstring(page.get_text('xml'))
        doc_json = {'id': doc_id, 'uid': get_uuid(), 'document': {}, 'label': '', 'img': {}}
        doc_json['label'] = label

        img_json = {
            "fname": '',
            "width": 0,
            "height": 0
        }
        WIDTH = float(root.get('width'))
        HEIGHT = float(root.get('height'))
        img_json['width'] = int(float(root.get('width')) * enlarge)
        img_json['height'] = int(float(root.get('height')) * enlarge)
        img_json['fname'] = os.path.basename(img_paths[pag_num])
        doc_json['img'] = img_json

        line_list = []
        for line_xml in root.xpath('//line'):
            line_json = {'id': '',
                         'linking': [],
                         'box': [],
                         'text': '',
                         'label': '',
                         'words': []}
            line_bbox = [int(float(v) * enlarge) for v in line_xml.get('bbox').split()]
            # line_bbox = valide_bbox(line_bbox, img_paths)
            line_json['box'] = line_bbox
            text = ''
            font = line_xml.getchildren()[0]
            word_list = []
            for char in font.getchildren():
                word_json = {'box': [], 'text': ''}
                token = char.get('c')
                text += token
                pos = [int(float(v) * enlarge) for v in char.get('quad').split()]
                token_bbox = [pos[0], pos[1], pos[-2], pos[-1]]
                # token_bbox = valide_bbox(token_bbox, img_paths)
                word_json['box'] = token_bbox
                word_json['text'] = token
                word_list.append(word_json)
            line_json['text'] = text
            line_json['id'] = line_id
            line_json['words'] = word_list
            line_list.append(line_json)
            line_id += 1
        # 从pdf中图片识别出line_list
        line_list_from_img = rec_img_from_doc(img_save_dir, WIDTH, HEIGHT, img_paths)
        line_list += line_list_from_img
        if line_list == []:
            continue
        doc_json['document'] = line_list
        documents['documents'] += [doc_json]
        doc_id += 1


def box_combine(boxes, overlap=50):
    '''
    行对齐
    :param boxes: [{box:[]}, {box:[]}]
    :param overlap: 最好设置低一些，这样行检测更准
    :return: [{box:[]}, {box:[]}] 排好序的
    '''

    # 先根据y值排序
    _boxes = [(t['box'][1], t) for t in boxes]
    sorted_boxes = sorted(_boxes, key=lambda x: x[0])
    boxes = [t[1] for t in sorted_boxes]

    high_upper_list = []
    high_lower_list = []
    width_left_list = []
    lines = []
    groups = []

    is_start = True
    last_row_lower = None

    for box in boxes:
        pos = box['box']
        high_upper, high_lower = pos[1], pos[3]
        width_left = pos[0]

        width_left_list.append(width_left)
        groups.append(box)

        if is_start:
            high_upper_list.append(high_upper)
            high_lower_list.append(high_lower)
            is_start = False

        average_high_upper = sum(high_upper_list) / len(high_upper_list)
        average_high_lower = sum(high_lower_list) / len(high_lower_list)
        coverage_percent = (average_high_lower - high_upper) / (average_high_lower - average_high_upper) * 100

        if coverage_percent > overlap:
            high_upper_list.append(high_upper)
            high_lower_list.append(high_lower)
        else:
            del width_left_list[-1]
            del groups[-1]

            if not last_row_lower:
                last_row_lower = average_high_lower
            else:
                last_row_lower = average_high_lower

            # 将一行的文本框根据x轴排序
            sorted_boxes = []
            for index in np.argsort(width_left_list):
                sorted_boxes.append(groups[index])

            lines.append(sorted_boxes)

            groups = []
            groups.append(box)
            width_left_list = []
            width_left_list.append(width_left)
            high_upper_list = []
            high_lower_list = []
            high_upper_list.append(high_upper)
            high_lower_list.append(high_lower)

    # 把最后的box保存下来
    if high_upper_list != []:
        # 将一行的文本框根据x轴排序
        sorted_boxes = []
        for index in np.argsort(width_left_list):
            sorted_boxes.append(groups[index])
        lines.append(sorted_boxes)
    # 处理结束
    tmp_lines = []
    for group in lines:
        tmp_lines.extend(group)
    return tmp_lines


def group_token(lines):
    '''
    将文档中所有无序的token bbox进行行对齐，文本块合并成segment，为layout提供先验信息
    :param lines:
    :return:
    '''
    all_data = []
    for line in lines:
        gaps = []
        last_token_x2 = 0
        for i, token in enumerate(line):
            if i == 0:
                last_token_x2 = token['pos'][2]
                continue
            gaps.append(token['pos'][2] - last_token_x2)
            last_token_x2 = token['pos'][2]

        segments = []
        last_gap = None
        tmp_segment = []
        for i, gap in enumerate(gaps):
            if i == 0:
                last_gap = gap
                tmp_segment.append(i)
                continue

            if abs(gap - last_gap) < 20:
                tmp_segment.append(i)
            else:
                segments.append(tmp_segment)
                tmp_segment = []
                tmp_segment.append(i)
            last_gap = gap
        if tmp_segment != []:
            segments.append(tmp_segment)

        data = []
        for seg in segments:
            data.append(line[seg[0]:seg[-1] + 2])
        all_data.append(data)
    return all_data


def align_segment():
    '''
    将documents中的文档取出做行对齐
    :return:
    '''
    global documents
    docs = documents['documents']
    new_docs = []
    for doc in docs:
        boxes = doc['document']
        sorted_boxes = box_combine(boxes)
        doc['document'] = sorted_boxes
        new_docs.append(doc)
    documents['documents'] = new_docs


def show_boxes(boxes, img_path, save_dir=None):
    '''
    将doc上的box绘出
    :param boxes:[{box:[], text:string}, {}]
    :param img_path:
    :return:
    '''
    print('图片路径:%s' % img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    for index, box in enumerate(boxes):
        bbox = box['box']
        text = box['text']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
        cv2.putText(img, str(index), (bbox[0], bbox[1]), font, 0.8, (0, 255, 0), 1)

    if save_dir is None:
        save_dir = './tmp'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)


def show_doc_boxes():
    '''
    展示doc bbox的结果，并把文本顺序输出
    :return:
    '''
    global documents
    docs = documents['documents']
    for doc in docs:
        img_path = os.path.join('images', doc['img']['fname'])
        show_boxes(doc['document'], img_path)


def main():
    data_path = 'data'
    save_dir = 'images'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('pdf'):
                pdf_path = os.path.join(root, file)

                # pdf -> pic
                img_paths = pyMuPDF_fitz(pdf_path, save_dir)

                # 识别图片，并和pdf文字解析结果合并
                doc_recognition(pdf_path, img_paths)

                # 将document中的line行对齐
                align_segment()

    show_doc_boxes()

    # 保存documents
    with open('data/zh.test.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False)

    # # 保存id2label
    # id2label = {i: label for i, label in enumerate(labels)}
    # with open('data/id2label.json', 'w', encoding='utf-8') as f:
    #     json.dump(id2label, f, ensure_ascii=False)

    # 保存labels
    with open('data/datasets_labels.json', 'w', encoding='utf-8') as f:
        json.dump({'labels': list(datasets_labels)}, f, ensure_ascii=False)

    print('length of documents is %d' % (doc_id))


if __name__ == '__main__':
    main()

    # doc_recognition('data\运单\运单帝国1.pdf', '')
