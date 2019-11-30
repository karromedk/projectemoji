from PIL import Image
import os

directory = '/Users/back/Documents/GitHub/emoji-stylegan/high_quality_data/'
c=1
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        print(filename)
        im = Image.open(directory+filename).convert('RGBA')
        im.load()
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])
        name=str(c)+'.jpg'
        background.save(name, 'JPEG', quality=100)
        
        #rgb_im.save(name)
        c+=1
        #os.remove(filename)
        continue
    else:
        continue