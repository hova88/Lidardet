"""
Code based on Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""
import torch 
from torch import nn 
from torch.nn import functional as F 
from libs.tools import get_paddings_indicator,change_default_args
from libs.nn import Empty 


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,128),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):

        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        return batch_canvas


#########################################################################
#Converting to tensorrt script requires fixed #batch(1)
#########################################################################


class PFNLayer_trt(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer

        if not self.last_vfe:
            out_channels = out_channels //2

        self.units = out_channels
        self.in_channels = in_channels

        if use_norm:
            BatchNorm2d = change_default_args(eps=1e-3,momentum=0.01)(nn.BatchNorm2d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm2d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels,self.units)
        self.norm = BatchNorm2d(self.units)


    def forward(self,inputs):
        x = self.linear(inputs)
        x = self.norm(x.permute(0,3, 1, 2).contiguous()).permute(0, 2,3,1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x,dim = 2,keepdim=True)[0]
        #print(x_max.shape)
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, 1,inputs.shape[2], 1)
            x = x.reshape_as(x_repeat)
            x_concatenated = torch.cat([x, x_repeat], dim=3)
            return x_concatenated


class PillarFeatureNet_trt(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,128),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0 
        num_input_features  += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        #pfn layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_channels = num_filters[i]
            out_channels = num_filters[i+1]
            if i < len(num_filters) -2:
                last_layer = False 
            else:
                last_layer = True 
            
            pfn_layers.append(PFNLayer_trt(in_channels,out_channels,use_norm,last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        # self.vx = voxel_size[0]
        # self.vy = voxel_size[1]
        # self.x_offset = self.vx / 2 + pc_range[0]
        # self.y_offset = self.vy / 2 + pc_range[1]
    

    def forward(self, pillar_x, pillar_y, pillar_z, pillar_i, num_voxels, x_sub_shaped, y_sub_shaped, mask):

        # Find distance of x, y, and z from cluster center
        # pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 3)
        pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 1)

        # points_mean = pillar_xyz.sum(dim=2, keepdim=True) / num_voxels.view(1,-1, 1, 1)
        points_mean = pillar_xyz.sum(dim=3, keepdim=True) / num_voxels.view(1, 1, -1, 1)
        f_cluster = pillar_xyz - points_mean
        # Find distance of x, y, and z from pillar center
        #f_center = torch.zeros_like(features[:, :, :2])
        #f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        #f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        f_center_offset_0 = pillar_x - x_sub_shaped
        f_center_offset_1 = pillar_y - y_sub_shaped

        f_center_concat = torch.cat((f_center_offset_0, f_center_offset_1), 1)

        pillar_xyzi = torch.cat((pillar_x, pillar_y, pillar_z, pillar_i), 1)
        features_list = [pillar_xyzi, f_cluster, f_center_concat]

        features = torch.cat(features_list, dim=1)
        masked_features = features * mask
        #print(masked_features.shape)
        masked_features = masked_features.permute(0,2,3,1)

        pillar_feature = self.pfn_layers[0](masked_features)
        return pillar_feature


class PointPillarsScatter_trt(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64,
                 batch_size=1):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features
        self.batch_size = 1

    # def forward(self, voxel_features, coords, batch_size):
    def forward(self, voxel_features, coords,voxel_mask):
        #voxel_feature = [max_voxel,64]
        
        # batch_canvas will be the final output.
        batch_canvas = []
        
        for batch_itt in range(self.batch_size):
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                     device=voxel_features.device)
            #print(voxel_mask.shape,'voxel_mask')

            this_coords = coords[voxel_mask[batch_itt]]
            
            indices = this_coords[:, 1] * self.nx + this_coords[:, 2]
            indices = indices.type(torch.float)

            # voxels = voxel_features[voxel_mask]
            voxels = voxel_features.t()[voxel_mask[batch_itt]]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            # canvas[:, indices] = voxels

            indices_2d = indices.view(1, -1)
            ones = torch.ones([self.nchannels, 1], dtype=torch.float, device=voxel_features.device)
            indices_num_channel = torch.mm(ones, indices_2d)
            indices_num_channel = indices_num_channel.type(torch.int64)
            canvas.scatter_(1, indices_num_channel, voxels)

            batch_canvas.append(canvas)
            
        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(self.batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas







if __name__ == '__main__':
    pass
    from torch2trt import torch2trt
    
    def generate_tensor_list(max_voxel_num,device,float_type):
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
    
    max_voxel_num = 12000
    example_tensor=generate_tensor_list(max_voxel_num,float_type=torch.float32,device='cuda')

    pfn = PFNLayer_trt(9,64).eval().cuda()
    spatial_features = torch.zeros((1,12000,100,9)).cuda()
    res = pfn(spatial_features)
    print(res.shape)
    trt = torch2trt(pfn,[spatial_features],fp16_mode=True,max_workspace_size=1 << 20)
    print('tensort======> pfn format success')

    PFN = PillarFeatureNet_trt().eval().cuda()
    res = PFN(*example_tensor)
    print(res.shape)
    trt = torch2trt(PFN,example_tensor,fp16_mode=True,max_workspace_size=1 << 20)
    print('tensort======> PFN format success') 