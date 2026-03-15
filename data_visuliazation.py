import matplotlib.pyplot as plt
import torch

torch.manual_seed(42)

def visulize(train_data,class_names,show=False):
    row = 4
    col = 3

    for i in range(1,row * col + 1):
        idx = torch.randint(0,len(train_data),size=[1]).item()
        img,label = train_data[idx]
        img = img.permute(1,2,0)
        plt.subplot(row,col,i)
        plt.imshow(img.squeeze() , cmap = "gray")
        plt.axis(False)
        plt.title(class_names[label])
    if show:
        plt.show()