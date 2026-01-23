def remap_checkpoints(state, target_model):
    """
    Remap checkpoint keys ONLY if the target model expects the new names.
    
    Args:
        state (dict): checkpoint['model'] state_dict
        target_model (nn.Module): diffusion model you are loading into
    """
    target_keys = set(target_model.state_dict().keys())
    if any('downs.' in k for k in target_keys):
        remapped = {}
        for k, v in state.items():
            nk = k
            nk = nk.replace('downs_label_noise.', 'downs.') 
            nk = nk.replace('final_conv.0.', 'final_conv.')
            remapped[nk] = v    
        return remapped
    else: 
        return state
