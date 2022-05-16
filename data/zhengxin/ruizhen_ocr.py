import requests
import os
import io
import traceback
import time


class RuizhenAngle:
    def __init__(self):
        self.url = 'http://172.25.128.169/service/ocr_gateway?expects=general-cn-v3_1'

    def request_in_image_file(self, image_file):
        json_data = {}
        assert (os.path.isfile(image_file))
        try:
            with io.open(image_file, 'rb') as f:
                image_bytes = f.read()
                json_data = self.request_in_image_bytes(image_bytes, {'image_path': image_file})
        except Exception as e:
            print("RuizhenOcr OCR recognition failed!!!")
            print("RuizhenOcr Exception:{}".format(str(e)))

        return json_data

    def request_in_image_bytes(self, image_bytes: bytes, cfg: dict):
        image_path = cfg['image_path']
        image_name = os.path.basename(image_path)
        result = {}
        try:
            headers = {}
            data_body = {}
            file_prefix, file_postfix = os.path.splitext(image_name)
            file_postfix = file_postfix.replace('.', '')
            files = [('image_file', (image_name, image_bytes, 'image/{}'.format(file_postfix)))]

            # post request
            r = requests.request("POST", url=self.url, headers=headers, data=data_body, files=files)
            if r.status_code != 200:
                print("failed to get info")
            else:
                result = r.json()
        except:
            traceback.print_exc()
        return result


def test_bytes_rec(b):
    ruizhen_ocr = RuizhenAngle()
    res = ruizhen_ocr.request_in_image_bytes(b, {'image_path':''})
    print(res)


if __name__ == '__main__':
    ruizhen_ocr = RuizhenAngle()
    res = ruizhen_ocr.request_in_image_file('./test_data/scoring_network.png')
    print(res)

    # img = cv2.imread('/work/Codes/layoutlmft/examples/XFUND-DATA-Gartner/zh.val/zh_val_0.jpg')
    # b = np.array(cv2.imencode('.png', img)[1]).tobytes()
    # test_bytes_rec(b)
