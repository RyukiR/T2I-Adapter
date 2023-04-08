import os
import cv2
import torch
from torch.cuda.amp import autocast
from model import get_sd_models
from adapter import get_adapters, get_adapter_feature
from condition import ExtraCondition
from diffusion import diffusion_inference
from utils import tensor2img, seed_everything
import api

opt = {
    'seed': 42,
    'outdir': 'outputs',
    'n_samples': 1,
    'cond_inp_type': 'image',
    'which_cond': 'cond1',
    'prompt': '',
    # 其他参数
}

# 准备模型
sd_model, sampler = get_sd_models(opt)
adapter = get_adapters(opt, getattr(ExtraCondition, opt['which_cond']))
cond_model = None
if opt['cond_inp_type'] == 'image':
    cond_model = get_cond_model(opt, getattr(ExtraCondition, opt['which_cond']))
process_cond_module = getattr(api, f'get_cond_{opt["which_cond"]}')

# 待处理的图片路径列表
image_paths = [
    'image1.png',
    'image2.png',
    'image3.png',
    # ...
]

# 对每张图片进行处理
for idx, image_path in enumerate(image_paths):
    # 加载图片并预处理
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cond = process_cond_module(opt, img, opt['cond_inp_type'], cond_model)

    # 获取特征向量和上下文
    adapter_features, append_to_context = get_adapter_feature(cond, adapter)

    # 执行扩散推断
    with torch.inference_mode(), sd_model.ema_scope(), autocast('cuda'):
        seed_everything(opt['seed'])
        result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)

    # 保存结果
    base_count = len(os.listdir(opt['outdir'])) // 2
    cv2.imwrite(os.path.join(opt['outdir'], f'{base_count:05}_cond.png'), tensor2img(cond))
    cv2.imwrite(os.path.join(opt['outdir'], f'{base_count:05}_result.png'), tensor2img(result))
