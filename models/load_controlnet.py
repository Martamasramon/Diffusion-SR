import re

def load_pretrained_with_controlnet(diffusion, ckpt, verbose=True):
    """
    Load a checkpoint from a model w/o ControlNet into a model w/ ControlNet by:
      - copying shared UNet weights into ControlNet's encoder + mid
      - keeping all ControlNet zero-convs at zero init
    """
    src = ckpt.get('model', ckpt)  # support {'model': state_dict} or plain state_dict

    # current params
    dst = diffusion.state_dict()
    out = {}

    def get_src_tensor(key):
        """Try 'key' then without leading 'model.' if present."""
        if key in src:
            return src[key]
        if key.startswith('model.') and key[6:] in src:
            return src[key[6:]]
        return None

    # regex helpers
    re_ctrl_prefix = re.compile(r'^(model\.)?controlnet\.(.+)$')

    skipped = []
    copied  = []
    missed  = []

    for k_dst in dst.keys():
        m = re_ctrl_prefix.match(k_dst)
        if not m:
            # non-controlnet param: load straight from ckpt if available, else keep current
            t = get_src_tensor(k_dst)
            if t is not None and t.shape == dst[k_dst].shape:
                out[k_dst] = t
                if verbose: copied.append(k_dst)
            else:
                out[k_dst] = dst[k_dst]
                if verbose and t is None:
                    missed.append(k_dst)
            continue

        # zero-convs must remain zero-initialized → skip copying
        if any(s in k_dst for s in [
            'controlnet.init_control_conv',   
            'controlnet.zero_convs',         
            'controlnet.mid_proj1',        
            'controlnet.mid_proj2',         
        ]):
            out[k_dst] = dst[k_dst]  
            if verbose: skipped.append(k_dst)
            continue

        # map controlnet to the equivalent unet path:
        if 'controlnet.init_main_conv' in k_dst:
            k_src = k_dst.replace('controlnet.init_main_conv', 'init_conv')
        elif '.downs.' in k_dst:
            k_src = k_dst.replace('controlnet.downs', 'downs_label_noise')
        elif any(s in k_dst for s in ['.mid_block1', '.mid_attn', '.mid_block2']):
            k_src = k_dst.replace('controlnet.', '')
        else:
            # anything else in controlnet.* that isn't a zero-conv – keep current
            out[k_dst] = dst[k_dst]
            if verbose: skipped.append(k_dst)
            continue

        # add "model." prefix counterpart as source
        k_src_full = k_src
        if not k_src_full.startswith('model.'):
            k_src_full = 'model.' + k_src

        # try to fetch from ckpt
        t = get_src_tensor(k_src_full)
        if t is None:
            t = get_src_tensor(k_src)  # try without 'model.' prefix

        if (t is not None) and (t.shape == dst[k_dst].shape):
            out[k_dst] = t
            if verbose: copied.append(f'{k_src_full if k_src_full in src else k_src} -> {k_dst}')
        else:
            # shape mismatch or missing → keep current (zeros or scratch)
            out[k_dst] = dst[k_dst]
            if verbose: missed.append(f'{k_src_full} for {k_dst}')

    diffusion.load_state_dict(out, strict=False)

    if verbose:
        print(f'Copied: {len(copied)} tensors')
        print(f'Skipped (kept init, e.g., zero-convs): {len(skipped)} tensors')
        if missed:
            print(f'Missed or shape-mismatch (kept current): {len(missed)}')
            # Uncomment for debugging:
            # for m in missed: print('  ', m)
