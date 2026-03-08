class ProcessOptions:

    def __init__(self, processordefines:dict, face_distance,  blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, show_face_area, show_mask=False, use_enhancer=True, blend_mode='seamless'):
        self.processors = processordefines
        self.face_distance_threshold = face_distance
        self.blend_ratio = blend_ratio
        self.face_swap_mode = swap_mode  # Cambiado de swap_mode a face_swap_mode para consistencia
        self.selected_index = selected_index
        self.masking_text = masking_text
        self.imagemask = imagemask
        self.num_swap_steps = num_steps
        self.show_face_area_overlay = show_face_area
        self.show_face_masking = show_mask
        # New options
        self.use_enhancer = use_enhancer
        self.blend_mode = blend_mode