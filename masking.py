import torch

def get_source_mask(source):
    size = source.size(1)
    src_pad = (source==2).nonzero()
    
    if len(src_pad) == 0:
        stop_idx = size
    else:
        stop_idx = src_pad[0][1].item()
        
    mask = source.clone()
    # Mask all padding
    mask[:,stop_idx:] = 1

    # Convert everything before stop_idx to zero
    mask[:,:stop_idx] = 0
    mask = mask.unsqueeze(0) > 0
    return mask


def get_target_mask(target):
    
    size=target.size(1)
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 1, float(0)).masked_fill(mask == 0, float(1))
    
    # Find out where the target pad starts
    trg_pad = (target==2).nonzero()

    # Check if there is no padding in sentence
    if len(trg_pad) == 0:
        stop_idx = size
    else:
        stop_idx = trg_pad[0][1].item()
        mask[stop_idx:, stop_idx:] = 1

    return mask.unsqueeze(0) > 0