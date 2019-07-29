import torch

batch_size = 3
height, width = 2, 2
min_ = 0.4
max_ = 0.5
feature_map = torch.randn(batch_size, height, width)
mask_map = torch.ones_like(feature_map)
for batch_idx in range(feature_map.size()[0]):
    mask = torch.ones_like(feature_map[batch_idx,:,:])
    max_mask = torch.ge(feature_map[batch_idx,:,:], max_)
    """
    Here both mask and max_mask are tensors of int dtype,so they couldn't set requires_grad = True; 
    the reason is that only tensors of floating point dtype can reqiure gradients; 
    """
    min_mask =  feature_map[batch_idx,:,:] < min_
    mask[max_mask.data] = 1     # .data is needed
    mask[min_mask.data] = -1
    mask_map[batch_idx,:,:] = mask

print(mask_map)
