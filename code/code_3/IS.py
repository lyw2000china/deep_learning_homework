from datasets import *
import numpy as np
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.nn import functional as F
import torch.utils.data
from scipy.stats import entropy
import torchvision.models


path = './picture'
count = 0
for root, dirs, files in os.walk(path):
    for each in files:
        count += 1
print('图片数量：', count)

batch_size = 16
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]

val_dataloader = DataLoader(
    ISImageDataset(path, transforms_=transforms_),
    batch_size=batch_size,
)

cuda = True if torch.cuda.is_available() else False
print('cuda: ', cuda)
tensor = torch.cuda.FloatTensor

# inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
inception_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1).cuda()

inception_model.eval()
up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).cuda()

def get_pred(x):
    if True:
        x = up(x)
    x = inception_model(x)
    return F.softmax(x, dim=1).data.cpu().numpy()

print('Computing predictions using inception v3 model')
preds = np.zeros((count, 1000))

for i, data in enumerate(val_dataloader):
    data = data.type(tensor)
    batch_size_i = data.size()[0]
    preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(data)

print('Computing KL Divergence')
split_scores = []
splits = 10
N = count
for k in range(splits):
    part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
    py = np.mean(part, axis=0)  # marginal probability
    scores = []
    for i in range(part.shape[0]):
        pyx = part[i, :]  # conditional probability
        scores.append(entropy(pyx, py))  # compute divergence
    split_scores.append(np.exp(np.mean(scores)))


mean, std = np.mean(split_scores), np.std(split_scores)
print('IS is %.4f' % mean)
print('The std is %.4f' % std)
