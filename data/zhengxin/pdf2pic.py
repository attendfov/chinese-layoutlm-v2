import fitz
import os
import uuid

# from angle.predict import angle_predict
# import rotate_img

def get_uuid():
    return ''.join(str(uuid.uuid4()).split('-'))


def pyMuPDF_fitz(pdfPath, save_dir, define_sav_img_path=True, topk=None, scale=4):
    basename = os.path.splitext(os.path.basename(pdfPath))[0]
    filename = '##'.join(os.path.dirname(pdfPath).split('\\'))
    # basename = filename + '-' + basename

    pdfDoc = fitz.open(pdfPath)
    zoom_x = scale  # (1.33333333-->1056x816)   (2-->1584x1224)
    zoom_y = scale
    img_paths = []
    if pdfDoc.pageCount == 1:  # 如果pdf只有一页
        for pg in range(pdfDoc.pageCount):
            page = pdfDoc[pg]
            rotate = int(0)
            # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
            # 此处若是不做设置，默认图片大小为：792X612, dpi=96
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = os.path.join(save_dir, basename, basename + '_1.png')
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        if define_sav_img_path:
            basename = get_uuid()
            img_path = os.path.join(save_dir, basename + '.jpg')
        pix.save(img_path)  # 将图片写入指定的文件夹内
        img_paths.append(img_path)
    else:
        k = 0
        for pg in range(pdfDoc.pageCount):
            page = pdfDoc[pg]
            rotate = int(0)
            # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
            # 此处若是不做设置，默认图片大小为：792X612, dpi=96
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_path = os.path.join(save_dir, basename, basename + '_%d.png' % pg)
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            if define_sav_img_path:
                basename = get_uuid()
                img_path = os.path.join(save_dir, basename + '.jpg')
            pix.save(img_path)
            img_paths.append(img_path)

            k += 1
            if topk and topk == k:
                break

            # if pix.h < pix.w:
            #     angle = angle_predict(path=img_path)
            #     rotate_img.rotate_bound(img_path, img_path, angle)
    return img_paths


if __name__ == "__main__":
    data_path = r'data\批2\data\新疆邮政_征信测试'
    save_dir = r'data\批2\标注'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('pdf'):
                pdf_path = os.path.join(root, file)
                pyMuPDF_fitz(pdf_path, save_dir, define_sav_img_path=False, topk=2, scale=4)

    # 生成样例
    # data_path = r'data\批2\data\新疆邮政_征信测试'
    # save_dir = r'data\批2\sample\scale6'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # k = 0
    # for root, dirs, files in os.walk(data_path):
    #     for file in files:
    #         if file.endswith('pdf'):
    #             pdf_path = os.path.join(root, file)
    #             pyMuPDF_fitz(pdf_path, save_dir, define_sav_img_path=False, topk=2, scale=6)
    #             k += 1
    #             if k == 2:
    #                 break
