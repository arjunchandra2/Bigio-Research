"""
Annotation Class for a set of annotations on a single subimage in a single .tif image
"""

class Annotation:

    #list of all annotations for a single .tif image
    annotations = []

    def __init__(self, bboxes, cls_names, confs, z_plane, window_left, window_upper) -> None:
        self.bboxes = bboxes
        self.cls_names = cls_names
        self.confs = confs
        self.z_plane = z_plane
        self.window_left = window_left
        self.window_upper = window_upper

        Annotation.annotations.append(self)
    
        
    def __str__(self) -> str:
        return f"z_plane: {self.z_plane} \nBboxes: {self.bboxes} \nClasses: {self.cls_names}\nConfidences: {self.confs} "
