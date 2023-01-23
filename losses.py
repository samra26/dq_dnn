import torch

def hausdorff(x, y):
 
    x = x.float()
    y = y.float()
    #print(x.shape,y.shape)
    distance_tensor = torch.cdist(x, y, p=2)
    #print(distance_tensor.shape)
    
    value1 = distance_tensor.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_tensor.min(1)[1].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)
    value =  value.max(1)[0]
    
    return value.mean()
