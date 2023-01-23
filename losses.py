import torch

def hausdorff_2d_torch(x, y):
    x=x.squeeze()
    y=y.squeeze()
    x = x.float()
    y = y.float()
    
    distance_tensor = torch.cdist(x, y, p=2)
    
    
    value1 = distance_tensor.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_tensor.min(1)[1].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]
