import requests

url = 'http://10.86.20.6:10004/kv_extract'
files = {'file': open(r'D:\Codes\Doc-understanding\layoutlmft\examples\requests_images\57d5b17daf0546da92a8ec0d4e255f85.jpg', 'rb')}
# files = {'file': ('report.jpg', open('/home/test.mpg', 'rb'))}     #显式的设置文件名

r = requests.post(url, files=files)
print(r.content)





# url = 'http://10.86.20.6:10004/result/c11150b7d38e4fa384b8de49f877c151.jpg'
# r = requests.get(url)
# print(r.content)