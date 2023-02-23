import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def viz_results(epoch, trainloader, size):

    data = next(iter(trainloader))
    with open('grads_' +str(epoch)+'.npy', 'rb') as f:
        grads = np.load(f, allow_pickle=True)
        
    #num = random.randint(0, len(data))
    num = 7
    if len(size) == 3:
        pic = torch.permute(data[0][num],(1,2,0))
    else :
        pic  = np.reshape(data[0][num],size)
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(pic)
    plt.axis('off')
    plt.subplot(3,1,2)
    plt.imshow(grads[0][num], cmap ='gray')
    plt.axis('off')
    plt.subplot(3,1,3)
    plt.imshow(pic)
    plt.imshow(grads[0][num], 'jet', interpolation='none', alpha=0.5)
    plt.axis('off')
    plt.savefig(str(epoch)+str(num)+'.png')
    plt.show()
    plt.close()
    
    means = []
    for i in range(len(grads)):
      for j in range(len(grads[i])):
        temp = np.mean(grads[i][j])
        means.append(temp)
        
    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(means)
    plt.savefig('box_'+str(epoch)+'.png')
    plt.show()