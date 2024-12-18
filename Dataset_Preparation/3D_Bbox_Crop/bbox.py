"""
Bounding Box Class 
"""

class Bbox:

    #threshold for bbox cropping
    OVERLAP_THRESHOLD = 0.8

    #count of number of bboxes removed as a result of checking annotations 
    BBOXES_REMOVED = 0

    #maintain a list of all unseen bounding boxes 
    bboxes_unseen = {}

    #keep track of last bbox loaded to ensure data is clean/annotations are not jumping
    last_bbox = None

    
    def __init__(self, top_left_x, top_left_y, width, height, z_plane, class_name, conf=1.0) -> None:
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.width = width
        self.height = height
        self.z_plane = z_plane
        self.class_name = class_name
        self.conf = conf

        #update Class variables 
        if z_plane in Bbox.bboxes_unseen:
            #do not add duplicates from annotation file
            if self not in Bbox.bboxes_unseen[z_plane]:
                Bbox.bboxes_unseen[z_plane].append(self)
                #self.check()
            else:
                #removed a duplicate
                Bbox.BBOXES_REMOVED += 1 
        else:
            Bbox.bboxes_unseen[z_plane] = [self]
            #self.check()
            
        Bbox.last_bbox = self

    
    
    #this method is unused for now (3/19) until we decide about cleaning data manually 
    def check(self):
         #remove annotation if it is not clean - (exact same bbox coords jumps multiple z-planes)
        if Bbox.last_bbox is not None:
            #explicit cast to int to avoid OverflowError - not sure why
            if self.coords_equal(Bbox.last_bbox) and abs(int(self.z_plane)-int(Bbox.last_bbox.z_plane)) >= 2:
                print('REMOVING:')
                print(self)
                print('BECUASE LAST WAS:')
                print(Bbox.last_bbox)
                Bbox.bboxes_unseen[self.z_plane].remove(self)
                Bbox.BBOXES_REMOVED += 1


   
    def center_x(self):
        return self.top_left_x + self.width//2
    
    def center_y(self):
        return self.top_left_y + self.height//2

    def get_coords(self):
        return self.top_left_x, self.top_left_y, self.width, self.height

    def coords_equal(self, other):
        return self.top_left_x == other.top_left_x and self.top_left_y == other.top_left_y and self.width == other.width \
        and self.height == other.height

    
    def __eq__(self, other) -> bool:
        if isinstance(other, Bbox):
            return self.top_left_x == other.top_left_x and self.top_left_y == other.top_left_y and self.width == other.width \
            and self.height == other.height and self.z_plane == other.z_plane

        return NotImplemented
    

    def __str__(self) -> str:
        return "Class: " + self.class_name + "\n" + "z_plane: " + str(self.z_plane) + "\n" + \
         "Bbox: [" + str(self.top_left_x)  + ", " + str(self.top_left_y) + ", " + str(self.width) + \
        ', ' + str(self.height) + "] \n"
        
        
        
        
        