from PIL import Image

def get_img(path):
    return Image.open(path).convert('RGB')

img_path = '/home/fengyuan/JittorModels/trainingSet/trainingSet/0/img_1.jpg'

a = get_img(img_path)
print(a.shape)