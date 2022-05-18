import os
import json
import uuid
import cv2
import copy

from ruizhen_ocr import RuizhenAngle
from test_cyclone_ocr import cyclone_ocr

ruizhen_ocr = RuizhenAngle()

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

data_labels = {'1': ['100', '.100'], '2': ['200', '201', '200.201'],
               '3': ['300', '.300', '301', '.301', '302', '.302', '303', '.303'],
               '4': ['400', '.400', '401', '.401', '405', '.405', '406', '.406'],
               '5': ['500', '501', '500.501', '502', '500.502', '503', '500.503']}

all_labels = ['100', '.100', '200', '201', '200.201', '300', '.300', '301', '.301', '302', '.302', '303', '.303',
              '400', '.400', '401', '.401', '405', '.405', '406', '.406', '500', '501', '500.501', '502', '500.502',
              '503', '500.503']

labels = ['.100', '200.201', '200.202', '.300', '.301', '.302', '.303',
          '.400', '.401', '.405', '.406', '500.501', '500.502', '500.503']


def get_uuid():
    return ''.join(str(uuid.uuid4()).split('-'))


def cal_score(anno_region, ocr_region):
    flag = 0
    (a_x0, a_y0), (a_x1, a_y1) = anno_region['points']
    [o_x0, o_y0, o_x1, o_y1] = ocr_region['box']
    x0 = max(a_x0, o_x0)
    x1 = min(a_x1, o_x1)
    y0 = max(a_y0, o_y0)
    y1 = min(a_y1, o_y1)
    joint_area = max(x1 - x0, 0) * max(y1 - y0, 0)
    if joint_area <= 0:
        return 0, flag
    anno_area = (a_x1 - a_x0) * (a_y1 - a_y0)
    ocr_area = (o_x1 - o_x0) * (o_y1 - o_y0)
    iou_score = joint_area / (anno_area + ocr_area - joint_area)

    # 检测ocr多个区域对应一个标注区域
    if joint_area / anno_area < 0.6:
        flag = 1
    return iou_score, flag


def merge_ocr_box(document, merge_indexs):
    a = copy.deepcopy(merge_indexs)
    lines = []
    all_index = []
    for indexs in merge_indexs:
        label, anno_points = indexs.pop()
        tmp = []
        for index in indexs:
            all_index.append(index)
            tmp.append(document[index])
        try:
            pos = indexs[0]
        except:
            print(3)
        lines.append((tmp, label, anno_points, pos))

    new_document = []
    for j in range(len(document)):
        if j not in all_index:
            new_document.append(document[j])

    for segs in lines:
        lls, label, anno_points, pos = segs
        merge_res = lls.pop(0)
        merge_res['label'] = label
        merge_res['box'] = [*anno_points[0], *anno_points[1]]
        for li in lls:
            merge_res['text'] += li['text']
            words = merge_res['words']
            words.extend(li['words'])
            merge_res['words'] = words
        new_document.insert(pos, merge_res)
    return new_document


def align_annotation(doc_json, annotation):
    document = doc_json['document']

    merge_index = []
    for anno in annotation['shapes']:
        scores = []
        flag_all = 0
        for line in document:
            score, flag = cal_score(anno, line)
            scores.append(score)
            flag_all += flag

        # flag>=2时，说明一个标注区域对饮多个ocr识别框
        if flag_all >= 2:
            indexs = []
            for i, v in enumerate(scores):
                assert v < 0.6
                if v > 0.1:  # 过滤一些不符合的
                    indexs.append(i)
            if len(indexs) < 2:
                continue
                raise Exception('exception !')
            indexs.append((anno['label'], anno['points']))
            merge_index.append(indexs)
        else:
            max_score = max(scores)
            if max_score > 0.3:
                index = scores.index(max_score)
                document[index]['label'] = anno['label']
            elif max_score == 0:
                continue
            else:
                print(doc_json['img'])
                # raise Exception('未找到在原图像中的位置')
    # merge ocr box
    document = merge_ocr_box(document, merge_index)
    # 重新索引
    for i in range(len(document)):
        document[i]['id'] = i
    doc_json['document'] = document


def add_relation(doc_json):
    document = doc_json['document']
    line_index_with_label = {'1': [], '2': [], '3': [], '4': [], '5': []}
    for i, line in enumerate(document):
        l = line['label']
        if l != '':
            if l.startswith('1') or l.startswith('.1'):
                line_index_with_label['1'].append((l, i))
            elif l.startswith('2') or l.startswith('.2'):
                line_index_with_label['2'].append((l, i))
            elif l.startswith('3') or l.startswith('.3'):
                line_index_with_label['3'].append((l, i))
            elif l.startswith('4') or l.startswith('.4'):
                line_index_with_label['4'].append((l, i))
            elif l.startswith('5') or l.startswith('.5'):
                line_index_with_label['5'].append((l, i))
            else:
                raise Exception('unexpected label: %s' % l)

    for _, v in line_index_with_label.items():
        values = [_v for _v in v if '.' in _v[0]]
        keys = [_v for _v in v if '.' not in _v[0]]
        keys_text = [f[0] for f in keys]
        keys_index = [f[1] for f in keys]
        for single_v in values:
            d, index = single_v
            ds = [a for a in d.split('.') if a != '']
            for g in ds:
                if g in keys_text:
                    inde = keys_index[keys_text.index(g)]
                    # index时value的索引，inde是key的索引
                    relation = sorted([index, inde])
                    document[index]['linking'].append((relation[0], relation[1]))
                    document[inde]['linking'].append((relation[0], relation[1]))
    doc_json['document'] = document


def format_data_to_xfund(doc_id, img_path, annotation, ocr_res):
    doc_json = {'id': doc_id, 'uid': get_uuid(), 'document': [], 'label': '', 'img': {"fname": img_path}}
    line_id = 0
    for line in ocr_res:
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
        doc_json['document'] += [line_json]
    if annotation != {}:
        align_annotation(doc_json, annotation)
        # add relation
        add_relation(doc_json)
    documents['documents'] += [doc_json]
    doc_id += 1


def ocr(data, save_path):
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            return json.load(f)['results']
    from tqdm import tqdm
    for i in tqdm(range(len(data))):
        img_path = data[i][1]
        # rec_res = cyclone_ocr(img_path)
        rec_res = ruizhen_ocr.request_in_image_file(img_path)
        assert len(rec_res['response']['data']['identify_results']) == 1
        data[i].append(rec_res['response']['data']['identify_results'][0]['details']['text2'])

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({'results': data}, f)
    return data


def read(data_path):
    data = []
    img_id = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('png'):
                img_path = os.path.join(root, file)
                annotation_path = os.path.join(root, os.path.splitext(file)[0] + '.json')
                if not os.path.exists(annotation_path):
                    data.append([img_id, img_path, {}])
                    continue
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation = json.load(f)
                    del annotation['imageData']
                data.append([img_id, img_path, annotation])
                img_id += 1
    print('nums of images is %d' % len(data))
    return data


def read_img(img_path):
    import cv2
    import numpy as np
    im = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  # 只能用该方法读取含有中文路径的图片
    return im


def save_img(img, img_path):
    import cv2

    cv2.imencode('.jpg', img)[1].tofile(img_path)  # 英文或中文路径均适用


def re_visualize():
    train_json = 'data/批1/train.json'
    output_path = 'data/批1/re-visualize'

    with open(train_json, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    for document in documents['documents']:
        img_path = document['img']['fname']
        img = read_img(img_path)

        for line in document['document']:
            if line['label'] != '':
                x0, y0, x1, y1 = line['box']
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

        for line in document['document']:
            if line['linking'] != []:
                for head, tail in line['linking']:
                    head_box = document['document'][head]['box']
                    head_center = (int((head_box[0] + head_box[2]) / 2), int((head_box[1] + head_box[3]) / 2))
                    tail_box = document['document'][tail]['box']
                    tail_center = (int((tail_box[0] + tail_box[2]) / 2), int((tail_box[1] + tail_box[3]) / 2))
                    cv2.line(img, head_center, tail_center, (0, 255, 0), 2)
        base_file = '/'.join(img_path.split('/')[-2:])
        img_save_path = os.path.join(output_path, base_file)
        if not os.path.exists(os.path.dirname(img_save_path)):
            os.makedirs(os.path.dirname(img_save_path))
        save_img(img, img_save_path)


def train_test_split(document):
    label2data = {}
    for i, doc in enumerate(document):
        for line in doc['document']:
            label = line['label']
            if label in labels:
                if label in label2data.keys():
                    label2data[label].append(i)
                else:
                    label2data[label] = [i]
    label2data = {k: list(set(v)) for k, v in label2data.items()}
    label2data = dict(sorted(label2data.items(), key=lambda x: len(x[1])))
    print('')

    train, dev = [], []
    dev_index = [97, 135, 8, 140, 142, 15, 156, 31, 71, 9, 134, 136, 32, 33, 1, 2, 3, 4]
    for doc in document:
        if doc['id'] in dev_index:
            dev.append(doc)
        else:
            train.append(doc)

    for i in range(len(train)):
        train[i]['id'] = i
    for i in range(len(dev)):
        dev[i]['id'] = i
    return train, dev


def split_train_data():
    '''
    根据label进行dev的获取
    :return:
    '''
    root = 'data/标注批1部分批2'
    train_json = 'data/标注批1部分批2/train.json'
    with open(train_json, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    document = documents['documents']
    for i in range(len(document)):
        document[i]['id'] = i

    import shutil

    train, dev = train_test_split(document)

    if not os.path.exists(os.path.join(root, 'zh.train')):
        os.mkdir(os.path.join(root, 'zh.train'))
    if not os.path.exists(os.path.join(root, 'zh.val')):
        os.mkdir(os.path.join(root, 'zh.val'))

    for i, doc in enumerate(train):
        img_path = doc['img']['fname']
        base_dest_path = os.path.join('zh.train', 'img_%d' % i + os.path.splitext(img_path)[1])
        dest_img_path = os.path.join(root, base_dest_path)
        shutil.copy(img_path, dest_img_path)
        train[i]['img']['fname'] = base_dest_path

    for i, doc in enumerate(dev):
        img_path = doc['img']['fname']
        base_dest_path = os.path.join('zh.val', 'img_%d' % i + os.path.splitext(img_path)[1])
        dest_img_path = os.path.join(root, base_dest_path)
        shutil.copy(img_path, dest_img_path)
        dev[i]['img']['fname'] = base_dest_path

    # label整理
    for i, doc in enumerate(train):
        for j, line in enumerate(doc['document']):
            if '.' not in line['label']:
                train[i]['document'][j]['label'] = 'other'

    for i, doc in enumerate(dev):
        for j, line in enumerate(doc['document']):
            if '.' not in line['label']:
                dev[i]['document'][j]['label'] = 'other'

    with open(os.path.join(root, 'zh.train.json'), 'w', encoding='utf-8') as f:
        documents['documents'] = train
        json.dump(documents, f)

    with open(os.path.join(root, 'zh.val.json'), 'w', encoding='utf-8') as f:
        documents['documents'] = dev
        json.dump(documents, f)


def main():
    data_path = 'data/标注批1部分批2/data'
    save_ocr_path = 'data/标注批1部分批2/ocr_rec_results.json'

    data = read(data_path)
    data_ocr = ocr(data, save_path=save_ocr_path)

    for doc_id, img_path, annotation, ocr_res in data_ocr:
        format_data_to_xfund(doc_id, img_path, annotation, ocr_res)

    # 保存
    save_train_json = 'data/标注批1部分批2/train.json'
    with open(save_train_json, 'w', encoding='utf-8') as f:
        json.dump(documents, f)


if __name__ == '__main__':
    # main()

    # re_visualize()

    split_train_data()
