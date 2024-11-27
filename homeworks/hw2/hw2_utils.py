def rescale_tensor(tensor, old_range, new_range):
    old_min, old_max = old_range
    new_min, new_max = new_range
    rescaled_tensor = new_min + (tensor - old_min) * (new_max - new_min) / (old_max - old_min)
    return rescaled_tensor