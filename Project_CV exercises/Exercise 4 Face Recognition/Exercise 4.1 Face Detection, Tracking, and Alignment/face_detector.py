import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size
        self.aligned = None

    	###########################ToDo: Specify all parameters for template matching.###########################
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size
        self.last_position = None
        self.first_frame = True
        self.rect = None
        
    
    def get_search_window(self,frame, last_position, padding=20):
        x, y, w, h = last_position
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        return frame[y_start:y_end, x_start:x_end], (x_start, y_start)
    
        

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, ret, image, template, threshold = 0.55):
        
        if ret:
            if template is None:
                faces = self.detect_face(image)
                if faces:
                    x, y, width, height = faces['rect']
                    self.last_position = (x, y, width, height)
                    self.rect = self.last_position
                    template = image[y:y+height, x:x+width]
                    self.aligned = self.align_face(image, self.rect)
            else:
                search_window, offset = self.get_search_window(image, self.last_position)
                res = cv2.matchTemplate(search_window, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
                if max_val < threshold:
                    template = None
                else:
                    x, y = max_loc
                    x += offset[0]
                    y += offset[1]
                    face_rect =x, y, 
                    self.last_position = (x, y, self.last_position[2], self.last_position[3])
                    self.rect = self.last_position
                    self.aligned = self.align_face(image, self.rect)
    
            return {"rect": self.rect, "image": image, "aligned": self.aligned, "response": 0}, template
            
        else:
            return {"rect": self.rect, "image": image, "aligned": self.aligned, "response": 0}, template
        
        
    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]
        
        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))
    

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
    

