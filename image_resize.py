from PIL import Image
import os, sys

path_in = "Z:/transfert_test/train_real/images/"
path_out = "Z:/transfert_test/train_real/images_resized/" 
dirs = os.listdir(path_in)

for item in dirs:
    if os.path.isfile(path_in+item):
        im = Image.open(path_in+item)
        f, e = os.path.splitext(path_in+item)
        imResize = im.resize((1280,760), Image.NEAREST)
        imResize.save(path_out + item[:-4] + '_resized.jpg', 'JPEG')
        #print(path_out + item + '_resized.png')
