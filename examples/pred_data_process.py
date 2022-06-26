import numpy as np
import os
import json


def align_bbox(input_path, save_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    for n, doc in enumerate(data["documents"]):
        document = doc["document"]

        boxes = []
        for line in document:
            bbox = line['box']
            boxes.append((bbox[1], bbox, line))
        boxes = sorted(boxes, key=lambda x: x[0])

        # 行检测
        groups = []
        group = []
        for _, bbox, line in boxes:
            if group == []:
                group.append((bbox, line))
            else:
                avg_y1 = 0
                avg_y2 = 0
                for box, _ in group:
                    avg_y1 += box[1]
                    avg_y2 += box[3]
                avg_y1 = avg_y1 / len(group)
                avg_y2 = avg_y2 / len(group)

                y1 = bbox[1]
                y2 = bbox[3]

                overlap = (avg_y2 - y1) / (y2 - avg_y1) * 100
                if overlap > 30:
                    group.append((bbox, line))
                else:
                    groups.append(group)
                    group = [(bbox, line)]
        if group != []:
            groups.append(group)

        # 行内bbox排序
        boxes = []
        for group in groups:
            group = sorted(group, key=lambda x: x[0][0])
            for box, line in group:
                # print(line['text'])
                boxes.append(line)

        data['documents'][n]['document'] = boxes

    with open(save_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == '__main__':
    '''
    在进行实体识别前，对bbox进行 行对齐
    '''
    json_path = '../DATA/xfund-and-funsd/XFUND-and-FUNSD/zh.val.json'
    save_path = '../DATA/xfund-and-funsd/XFUND-and-FUNSD/zh.val.align.json'
    align_bbox(json_path, save_path)
