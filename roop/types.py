class Face(dict):
    def __init__(self, bbox, kps, det_score, embedding=None, normed_embedding=None, landmark_106=None, gender=None, age=None):
        super().__init__()
        self['bbox'] = bbox
        self['kps'] = kps
        self['det_score'] = det_score
        self['embedding'] = embedding
        self['normed_embedding'] = normed_embedding
        self['landmark_106'] = landmark_106
        self['gender'] = gender
        self['age'] = age
        # For compatibility with code that accesses as attributes
        self.bbox = bbox
        self.kps = kps
        self.det_score = det_score
        self.embedding = embedding
        self.normed_embedding = normed_embedding
        self.landmark_106 = landmark_106
        self.gender = gender
        self.age = age
  
class Frame:  
    def __init__(self, data):  
        self.data = data  
  
class FaceSet:  
    def __init__(self):  
        self.faces = []  
        self.img_path = None 
