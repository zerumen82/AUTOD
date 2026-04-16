import os
import torch

def check_unet_channels():
    """Verifica el archivo del modelo SVD Turbo para ver la configuracion de canales"""
    model_path = os.path.abspath("ui/tob/ComfyUI/models/diffusion_models/StableDiffusionTurbo/svd_xt.safetensors")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Modelo no encontrado en {model_path}")
        return False
    
    print(f"OK: Modelo encontrado en {model_path}")
    
    try:
        import safetensors.torch
        state_dict = safetensors.torch.load_file(model_path, device="cpu")
        
        # Imprimir las claves relevantes
        print("\n=== Estado del modelo ===")
        print(f"Total de keys: {len(state_dict.keys())}")
        
        # Buscar entradas relacionadas con canales
        in_channels = None
        out_channels = None
        encoder_channels = None
        
        for key in state_dict.keys():
            if "in_channels" in key.lower():
                in_channels = state_dict[key]
                print(f"  In channels: {in_channels}")
            elif "out_channels" in key.lower():
                out_channels = state_dict[key]
                print(f"  Out channels: {out_channels}")
            elif "encoder" in key.lower() and "channels" in key.lower():
                encoder_channels = state_dict[key]
                print(f"  Encoder channels: {encoder_channels}")
        
        # Buscar el primer conv_in
        first_conv_key = None
        for key in sorted(state_dict.keys()):
            if "conv_in" in key.lower():
                first_conv_key = key
                break
        
        if first_conv_key:
            weight = state_dict[first_conv_key]
            print(f"\n=== Conv_in detalles ===")
            print(f"  Tipo: {type(weight)}")
            print(f"  Shape: {weight.shape}")
            
            if len(weight.shape) == 4:
                out_channels, in_channels, kernel_h, kernel_w = weight.shape
                print(f"  Conv_in in_channels: {in_channels}")
                print(f"  Conv_in out_channels: {out_channels}")
                print(f"  Kernel: {kernel_h}x{kernel_w}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    check_unet_channels()
