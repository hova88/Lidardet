#coding=utf-8

import sys
import os
import pathlib
import pickle
import shutil
import time
from functools import partial

import fire
import numpy as np
import torch
from params.configs import cfg,cfg_from_yaml_file 


from core import build_voxel_generator,build_target_assigner,build_box_coder
from models import build_network



from torch2trt import torch2trt
import argparse
import numpy as np



def generate_tensor_list(max_voxel_num, device, float_type):
    pillar_x = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    pillar_y = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    pillar_z = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    pillar_i = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    num_points_per_pillar = torch.ones([1, max_voxel_num], dtype=float_type, device=device)
    x_sub_shaped = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    y_sub_shaped = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    mask = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    example_list = [pillar_x, pillar_y, pillar_z, pillar_i,
                    num_points_per_pillar,x_sub_shaped, y_sub_shaped, mask]
    return example_list



def convert(config_path,
            weights_file,
            trt_path,
            max_voxel_num = 12000):
    """train a VoxelNet model specified by a config file.
    """

    trt_path = pathlib.Path(trt_path)
    model_logs_path = trt_path / 'model_logs'
    model_logs_path.mkdir(parents=True,exist_ok=True)


    config_file_bkp = 'pipeline.config'
    shutil.copyfile(config_path,str(model_logs_path/config_file_bkp))
    shutil.copyfile(weights_file, str(model_logs_path/weights_file.split('/')[-1]))

    config = cfg_from_yaml_file(config_path, cfg)
    model_cfg = config.MODEL


    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = build_voxel_generator(config.VOXEL_GENERATOR)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = build_box_coder(config.BOX_CODER)
    target_assigner_cfg = config.TARGET_ASSIGNER
    target_assigner = build_target_assigner(target_assigner_cfg,
                                                    bv_range, box_coder)
    ######################
    # BUILD NET
    ######################
    model_cfg.XAVIER = True 
    net = build_network(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    # net_train = torch.nn.DataParallel(net).cuda()
    # print("num_trainable parameters:", len(list(net.parameters())))
    # for n, p in net.named_parameters():
    #     print(n, p.shape)


    state_dict=torch.load(weights_file)
    net.load_state_dict(state_dict,strict = False)
    net.eval()

    #tensorrt引擎路径
    pfn_trt_path=str(trt_path/"pfn.trt")
    backbone_trt_path = str(trt_path / "backbone.trt")

    #生成模型虚假输入数据用于编译tensorrt引擎

    example_tensor=generate_tensor_list(max_voxel_num,float_type=torch.float32,device='cuda')


    print('----------------------------------------------------------------------------')
    print("************ TensorRT: The PFN subnetwork is being transformed *************")
    print('----------------------------------------------------------------------------')
    pfn_trt = torch2trt(net.pfn, example_tensor, fp16_mode=True,
                        max_workspace_size=1 << 20)
    torch.save(pfn_trt.state_dict(), pfn_trt_path)

    print('------------------------------------------------------------------------------')
    print("******** TensorRT: The BackBone subnetwork(RPN) is being transformed *********")
    print('------------------------------------------------------------------------------')
    pc_range=np.array(config.VOXEL_GENERATOR.POINT_CLOUD_RANGE)
    vs=np.array(config.VOXEL_GENERATOR.VOXEL_SIZE)
    fp_size=((pc_range[3:]-pc_range[:3])/vs)[::-1].astype(np.int)
    rpn_input = torch.ones((1, 64, fp_size[1], fp_size[2]), dtype=torch.float32, device='cuda')
    rpn_trt = torch2trt(net.rpn, [rpn_input], fp16_mode=True, max_workspace_size=1 << 20)
    torch.save(rpn_trt.state_dict(), backbone_trt_path)

    print("Done!")





if __name__ == '__main__':

    fire.Fire()


# python libs/tools/convert2trt.py convert --config_path=./params/configs/pointpillars_kitti_car_xy16.yaml --weights_file=./params/weights/pointpillars/voxelnet-296960.tckpt --trt_path=/home/hova/Lidardet/params/TensorRT/