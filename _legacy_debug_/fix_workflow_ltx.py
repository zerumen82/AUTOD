# Fix for comfy_workflows_fixed.py
# Read the file
with open('roop/comfy_workflows_fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Simple replacements to fix the workflow
# Replace CheckpointLoaderSimple with UNETLoader reference  
old1 = '"1": {"inputs": {"ckpt_name": model_name}, "class_type": "CheckpointLoaderSimple"}'
new1 = '"1": {"inputs": {"unet_name": "ltx-video-0.9.5/transformer.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"}'

content = content.replace(old1, new1)

# Replace VAE reference from ["1", 2] to ["3", 0]
content = content.replace('"vae": ["1", 2]', '"vae": ["3", 0]')

# Add a new VAELoader node
old_node3 = '''        # 3: Positive prompt
        "3": {"inputs": {"clip": ["1", 1], "text": prompt}, "class_type": "CLIPTextEncode"},'''
        
new_node3 = '''        # 3: VAELoader - cargar VAE
        "3": {"inputs": {"vae_name": "ltx-video-0.9.5_vae.safetensors"}, "class_type": "VAELoader"},
        
        # 4: Positive prompt
        "4": {"inputs": {"clip": ["1", 1], "text": prompt}, "class_type": "CLIPTextEncode"},'''

content = content.replace(old_node3, new_node3)

# Update negative prompt node number (3 -> 4)
old_neg = '''        # 4: Negative prompt
        "4": {"inputs": {"clip": ["1", 1], "text": negative_prompt}, "class_type": "CLIPTextEncode"},'''
new_neg = '''        # 5: Negative prompt
        "5": {"inputs": {"clip": ["1", 1], "text": negative_prompt}, "class_type": "CLIPTextEncode"},'''

content = content.replace(old_neg, new_neg)

# Update LTXVImgToVideo node number (5 -> 6)
content = content.replace('"5": {"inputs": {', '"6": {"inputs": {')

# Update references to LTXVImgToVideo (5 -> 6)
content = content.replace('"samples": ["5", 0]', '"samples": ["6", 0]')

# Update VAEDecode node number (6 -> 7)
content = content.replace('"6": {"inputs": {"vae": ["3", 0]', '"7": {"inputs": {"vae": ["3", 0]')

# Update VHS_VideoCombine node number (7 -> 8) 
content = content.replace('"7": {"inputs": {"images": ["6"', '"8": {"inputs": {"images": ["7"')

# Update references from 4->5, 5->6 for the LTXV inputs
content = content.replace('"positive": ["3"', '"positive": ["4"')
content = content.replace('"negative": ["4"', '"negative": ["5"')

# Write the modified content back
with open('roop/comfy_workflows_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File modified successfully!")
