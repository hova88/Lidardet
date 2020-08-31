from collections import namedtuple

import numpy as np
import torch

from models.detectors.pointpillars import PointPillars, PointPillars_trt
from core import build_losses 


def build_network(model_cfg,voxel_generator,target_assigner):
    """build the lovely pointpillars instance."""
    pfn_num_filters = model_cfg.PILLAR_FEATURE_EXTRACTOR.num_filters
    grid_size = voxel_generator.grid_size  
    pc_range= voxel_generator.point_cloud_range
    voxel_size = voxel_generator.voxel_size

    # dense_shape = [1] + grid_size[::-1].tolist() + [pfn_num_filters[-1]] 
    dense_shape = [1] + grid_size[::-1].tolist() + [pfn_num_filters[-1]] 
    # print('dense_shape,',dense_shape)
    num_input_features = model_cfg.NUM_POINT_FEATURES
    if model_cfg.WITHOUT_REFLECTIVITY:
        num_input_features = 3

    if not model_cfg.XAVIER:
        # loss config
        loss_norm_type = model_cfg.LOSS.loss_norm_type
        pos_cls_weight = model_cfg.LOSS.pos_class_weight
        neg_cls_weight = model_cfg.LOSS.neg_class_weight
        encode_rad_error_by_sin = model_cfg.ENCODE_RAD_ERROR_BY_SIN
        direction_loss_weight = model_cfg.LOSS.direction_loss_weight    
        losses = build_losses(model_cfg.LOSS)
        cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses

        model_cfg.update({  
                            'pc_range' : pc_range,
                            'voxel_size' : voxel_size,
                            'pfn_num_filters'   : pfn_num_filters,
                            'num_input_features': num_input_features,
                            'loss_norm_type': loss_norm_type,
                            'pos_cls_weight': pos_cls_weight,
                            'neg_cls_weight': neg_cls_weight,
                            'direction_loss_weight': direction_loss_weight,
                            'cls_loss_ftor': cls_loss_ftor,
                            'loc_loss_ftor': loc_loss_ftor,
                            'cls_weight': cls_weight,
                            'loc_weight': loc_weight,
                            })
        model = PointPillars(output_shape = dense_shape,
                             model_cfg=model_cfg,
                             target_assigner= target_assigner)
    if model_cfg.XAVIER:
        model_cfg.update({'pfn_num_filters' :  pfn_num_filters,
                        'num_input_features': num_input_features,
                        'loss_norm_type': None,
                        'pos_cls_weight': None,
                        'neg_cls_weight': None,
                        'direction_loss_weight': None,
                        'cls_loss_ftor': None,
                        'loc_loss_ftor': None,
                        'cls_weight': None,
                        'loc_weight': None,})

        model = PointPillars_trt(output_shape = dense_shape,
                                 model_cfg=model_cfg,
                                 target_assigner= target_assigner)

    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


