import matplotlib.pyplot as plt
from torchvision import utils

def debug_visualize(input, index = 0):
    image = (input[index, [2, 1, 0]] + 1.) / 2.
    plt.imshow(image.detach().cpu().numpy().transpose((1,2,0)))