from PIL import Image
import os, sys
from torchvision import transforms
import random

import torch
import torch.nn.functional as F

import numpy


path = "images1024x1024/"
dirs = os.listdir( path )

size_range = [512, 576, 640, 704, 768] 

size_flag = {}

print(f"Loading images from {path}")

for size in size_range:
    dest_path = path.replace("1024",str(size))
    print(f'Added dest_path {dest_path}')
    size_flag[size]=False
    if not os.path.exists(dest_path):
            os.makedirs(dest_path)

norm = transforms.Normalize((127.5, 127.5,127.5),(127.5, 127.5,127.5))
to_t = transforms.ToTensor()
                           
def prepare_img(img, size):
    img = numpy.array(img).astype('float')
    img = (img - 127.5)/ 127.5
    minibatch = torch.tensor(img).unsqueeze_(0)
    minibatch = minibatch.permute(0, 3, 1, 2)
     
    minibatch = F.interpolate(minibatch, size=(size,size), mode='bilinear')
    minibatch = ((minibatch + 1) / 2)
    minibatch = minibatch.numpy().transpose((0, 2, 3, 1))
    minibatch = (minibatch * 255).astype('uint8')
    return minibatch[0]


for item in sorted(dirs):
    if os.path.isfile(path+item):
        if item.endswith("png"):
            orig_im = Image.open(path+item)

            no = int(item[-5-4:-4])
            if no % 1000 == 0:
                print(no)

            for size in size_range:

                dest_path = path.replace("1024",str(size))
                #print(f'{dest_path} {size}')

                if os.path.isfile(dest_path+item):
                    pass
                else:
                    if not size_flag[size]:
                        print(size, " starts at ", no)
                        size_flag[size]=True

                    im = prepare_img(orig_im, size)
                    im = Image.fromarray(im)
                    im.save(dest_path+item, 'PNG', quality=100)


