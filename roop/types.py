class Face(dict):
    def __init__(self, bbox, kps, det_score, embedding=None, normed_embedding=None, landmark_106=None, gender=None, age=None, face_img=None, **kwargs):
        super().__init__()
        self['bbox'] = bbox
        self['kps'] = kps
        self['det_score'] = det_score
        self['embedding'] = embedding
        self['normed_embedding'] = normed_embedding
        self['landmark_106'] = landmark_106
        self['gender'] = gender
        self['age'] = age
        self['face_img'] = face_img
        # For compatibility with code that accesses as attributes
        self.bbox = bbox
        self.kps = kps
        self.det_score = det_score
        self.embedding = embedding
        self.normed_embedding = normed_embedding
        self.landmark_106 = landmark_106
        self.gender = gender
        self.age = age
        self.face_img = face_img
        for k, v in kwargs.items():
            self[k] = v
            setattr(self, k, v)
  
class Frame:  
    def __init__(self, data):  
        self.data = data  
  
class FaceSet:
    def __init__(self, faces=None, name=None):
        self.faces = faces if faces is not None else []
        self.name = name
        self.img_path = None
