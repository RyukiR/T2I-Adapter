import os

import cv2
import torch
import numpy as np
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

torch.set_grad_enabled(False)

def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument(
        '--which_cond',
        type=str,
        required=True,
        choices=supported_cond,
        help='which condition modality you want to test',
    )
    opt = parser.parse_args()
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'../Dataset/Preprocessed_256/trn_stim_data-{which_cond}' # 条件图片输出路径 ！！！
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 加载预处理模型
    cond_model = None
    if opt.cond_inp_type == 'image':
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))

    process_cond_module = getattr(api, f'get_cond_{which_cond}_image_based')

    # preprocess
    with torch.inference_mode(), \
            autocast('cuda'):
        
        # 从.npy文件中读取数据 (路径请按实际情况调整), 并做针对OpenCV的处理
        train_data_dir = r"D:\\NSD Dataset\\subj01\\trn_stim_data.npy"
        value_multi_trial_data_dir = r"D:\\NSD Dataset\\subj01\\val_stim_multi_trial_data.npy"
        value_single_trial_data_dir = r"D:\\NSD Dataset\\subj01\\val_stim_single_trial_data.npy"
        cond_images = np.load(train_data_dir) # 选择条件图片路径 ！！！
        print("Input shape:\n", cond_images.shape)
        cond_images = (cond_images * 255).astype(np.uint8) # 将float32格式的数组转换为unit8格式的数组，转换过程会造成数据精度损失
        cond_images = np.transpose(cond_images, (0, 2, 3, 1)) # 将数组格式转化为适用于OpenCV处理的格式

        # 遍历每个图像数据, 并对图像进行进一步的处理和操作
        list_cond_images = []
        for cond_image in cond_images:
            # 进行预处理，得到处理后的条件图片
            cond_image = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
            cond = process_cond_module(opt, cond_image, opt.cond_inp_type, cond_model)

            # 存储条件图片（png格式）
            base_count = len(os.listdir(opt.outdir))
            cv2.imwrite(os.path.join(opt.outdir, f'{base_count:07}_{which_cond}.png'), tensor2img(cond))

            # 将GPU上的Tensor转移到CPU上，并存储
            list_cond_images.append(cond.cpu().numpy())
        print("png done")

        # 将图片存为numpy文件
        nparray_cond_images = np.array(list_cond_images)
        nparray_cond_images = nparray_cond_images.astype(np.float32) / 255
        np.save(f'preprocessed_outputs/trn_stim_data-{which_cond}.npy', nparray_cond_images) # numpy文件输出路径 ！！！
        print("numpy done")


if __name__ == '__main__':
    main()

