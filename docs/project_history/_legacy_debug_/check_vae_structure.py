from safetensors import safe_open

f = safe_open('ui/tob/ComfyUI/models/vae/ltx-video-0.9.1_vae.safetensors', framework='pt')
keys = list(f.keys())

# Get down_blocks indices
down_block_indices = set()
for k in keys:
    if 'down_blocks' in k:
        parts = k.split('.')
        idx = parts[parts.index('down_blocks')+1]
        down_block_indices.add(int(idx))
print(f'down_blocks indices: {sorted(down_block_indices)}')

# Get up_blocks indices
up_block_indices = set()
for k in keys:
    if 'up_blocks' in k:
        parts = k.split('.')
        idx = parts[parts.index('up_blocks')+1]
        up_block_indices.add(int(idx))
print(f'up_blocks indices: {sorted(up_block_indices)}')

# Check for conv_out in down_blocks.1
conv_out_keys = [k for k in keys if 'down_blocks.1.conv_out' in k]
print(f'down_blocks.1.conv_out keys: {conv_out_keys[:5]}')

# Check for conv.conv.bias in down_blocks.1
conv_keys = [k for k in keys if 'down_blocks.1.conv.' in k]
print(f'down_blocks.1.conv. keys: {conv_keys[:5]}')

# Check for conv_out.conv1 in down_blocks.1
conv_out_conv1_keys = [k for k in keys if 'down_blocks.1.conv_out.conv1' in k]
print(f'down_blocks.1.conv_out.conv1 keys: {conv_out_conv1_keys[:5]}')

# Get shapes of some key tensors
print('\nKey tensor shapes:')
for key in ['decoder.up_blocks.0.resnets.0.conv1.conv.weight', 
            'encoder.down_blocks.0.resnets.0.conv1.conv.weight',
            'encoder.down_blocks.3.resnets.0.conv1.conv.weight']:
    if key in keys:
        tensor = f.get_tensor(key)
        print(f'{key}: {tensor.shape}')

# Check for 0.9.1 specific key
print('\nChecking for LTX Video 0.9.1 specific keys:')
ltx_091_key = 'decoder.last_time_embedder.timestep_embedder.linear_1.weight'
if ltx_091_key in keys:
    print(f'Found {ltx_091_key} - This is LTX Video 0.9.1 VAE')
else:
    print(f'NOT found: {ltx_091_key}')
    
# Check for timestep_embedder keys
timestep_keys = [k for k in keys if 'timestep_embedder' in k or 'time_embedder' in k]
print(f'\nTimestep embedder keys: {timestep_keys[:10]}')

# Count resnets per block
print('\nResnets per down_block:')
for i in range(4):
    resnet_keys = [k for k in keys if f'down_blocks.{i}.resnets.' in k]
    resnet_indices = set()
    for k in resnet_keys:
        parts = k.split('.')
        idx = parts[parts.index('resnets')+1]
        resnet_indices.add(int(idx))
    print(f'  down_blocks.{i}: {len(resnet_indices)} resnets')

print('\nResnets per up_block:')
for i in range(4):
    resnet_keys = [k for k in keys if f'up_blocks.{i}.resnets.' in k]
    resnet_indices = set()
    for k in resnet_keys:
        parts = k.split('.')
        idx = parts[parts.index('resnets')+1]
        resnet_indices.add(int(idx))
    print(f'  up_blocks.{i}: {len(resnet_indices)} resnets')

# Check for conv_out in each down_block
print('\nconv_out in down_blocks:')
for i in range(4):
    conv_out_keys = [k for k in keys if f'down_blocks.{i}.conv_out' in k]
    print(f'  down_blocks.{i}: {len(conv_out_keys)} keys')

# Get channel sizes for each down_block
print('\nChannel sizes for encoder down_blocks:')
for i in range(4):
    key = f'encoder.down_blocks.{i}.resnets.0.conv1.conv.weight'
    if key in keys:
        tensor = f.get_tensor(key)
        print(f'  down_blocks.{i}: {tensor.shape[0]} channels')
    else:
        print(f'  down_blocks.{i}: no resnets')

# Get channel sizes for decoder up_blocks
print('\nChannel sizes for decoder up_blocks:')
for i in range(4):
    key = f'decoder.up_blocks.{i}.resnets.0.conv1.conv.weight'
    if key in keys:
        tensor = f.get_tensor(key)
        print(f'  up_blocks.{i}: {tensor.shape[0]} channels')
    else:
        print(f'  up_blocks.{i}: no resnets')

# Check conv_in and conv_out
print('\nConv_in/out shapes:')
for key in ['encoder.conv_in.conv.weight', 'encoder.conv_out.conv.weight', 'decoder.conv_in.conv.weight', 'decoder.conv_out.conv.weight']:
    if key in keys:
        tensor = f.get_tensor(key)
        print(f'  {key}: {tensor.shape}')

# Check down_blocks structure more carefully
print('\nDetailed down_blocks structure:')
for i in range(4):
    print(f'\n  down_blocks.{i}:')
    # Get all keys for this block
    block_keys = [k for k in keys if f'encoder.down_blocks.{i}.' in k]
    # Get unique subkeys
    subkeys = set()
    for k in block_keys:
        parts = k.split(f'encoder.down_blocks.{i}.')[1].split('.')
        if len(parts) >= 1:
            subkeys.add(parts[0])
    print(f'    Subkeys: {sorted(subkeys)}')
    
    # Check resnets
    resnet_keys = [k for k in block_keys if 'resnets' in k]
    if resnet_keys:
        resnet_indices = set()
        for k in resnet_keys:
            parts = k.split('.')
            idx = parts[parts.index('resnets')+1]
            resnet_indices.add(int(idx))
        print(f'    Resnets: {len(resnet_indices)} blocks')
    
    # Check conv_out
    conv_out_keys = [k for k in block_keys if 'conv_out' in k]
    if conv_out_keys:
        print(f'    Has conv_out: Yes')
        # Get shape
        for k in conv_out_keys:
            if 'conv1.conv.weight' in k:
                tensor = f.get_tensor(k)
                print(f'      conv_out.conv1 shape: {tensor.shape}')
                break
    else:
        print(f'    Has conv_out: No')

# Check decoder up_blocks structure
print('\nDetailed up_blocks structure:')
for i in range(4):
    print(f'\n  up_blocks.{i}:')
    # Get all keys for this block
    block_keys = [k for k in keys if f'decoder.up_blocks.{i}.' in k]
    # Get unique subkeys
    subkeys = set()
    for k in block_keys:
        parts = k.split(f'decoder.up_blocks.{i}.')[1].split('.')
        if len(parts) >= 1:
            subkeys.add(parts[0])
    print(f'    Subkeys: {sorted(subkeys)}')
    
    # Check resnets
    resnet_keys = [k for k in block_keys if 'resnets' in k]
    if resnet_keys:
        resnet_indices = set()
        for k in resnet_keys:
            parts = k.split('.')
            idx = parts[parts.index('resnets')+1]
            resnet_indices.add(int(idx))
        print(f'    Resnets: {len(resnet_indices)} blocks')
    
    # Check upsamplers
    upsample_keys = [k for k in block_keys if 'upsamplers' in k]
    if upsample_keys:
        print(f'    Has upsamplers: Yes')
    else:
        print(f'    Has upsamplers: No')
