import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 准备真实数据分布和生成模型的图像数据
real_images_folder = './realdata/train_cifar10'
generated_images_folder = './picture'

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1).cuda()
inception_model.eval()

# 定义图像变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 计算FID距离值
# fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
#                                                  inception_model,
#                                                  transform = transform)
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], batch_size=32, device='cuda', dims=2048, num_workers=0)
print('FID value:', fid_value)

