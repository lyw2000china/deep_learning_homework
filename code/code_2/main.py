from Unet_ddpm import UNetModel
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def marginal_prob_std(t, sigma):
    # 计算p (x(t) | x(0))的标准差
    t = torch.as_tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    # 计算SDE的扩散系数.
    return torch.as_tensor(sigma ** t, device=device)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    return loss

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._upgrade(model, update_fn=lambda e, m: m)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    score_model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1
    )
    score_model.to(device)

    n_epochs = 101
    resume_epoch = 0
    batch_size = 32
    lr = 1e-4

    # 若模型已预训练，加载预训练模型参数
    if os.path.exists('model/CIFAR10_score_model.pth'):
        score_model.load_state_dict(torch.load('model/CIFAR10_score_model.pth', map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load('model/CIFAR10_score_model.pth')['epoch']
    else:
        os.makedirs('model', exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = CIFAR10('./realdata', train=True, transform=transform, download=True)

    # transform = transforms.Compose([transforms.ToTensor(),])
    # dataset = MNIST('./realdata', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = Adam(score_model.parameters(), lr=lr)
    ema = EMA(score_model)
    for epoch in range(resume_epoch, n_epochs):
        total_loss = 0
        num_items = 0
        for step, (x, y) in enumerate(data_loader):
            x = x.to(device)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(score_model)
            total_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            if step % 200 == 0:
                print('[%d/%d][%d/%d] Loss: %.4f'
                      % (epoch + 1, n_epochs, step, len(data_loader), loss.item()))

        print('[%d/%d] avg_Loss: %.4f' % (epoch + 1, n_epochs, total_loss / num_items))
        torch.save({'epoch': epoch + 1,
                    'state_dict': score_model.state_dict()},
                     'model/mnist_score_model.pth')
        print('epoch' + str(epoch + 1) + 'have saved')