import argparse
import cv2
from face_detector import FaceDetector

# The training module of the face recognition system. In summary, training comprises the following workflow:
#   1) Capturing new video frame.
#   2) Run face detection / tracking.
#   3) Extract face embedding and update face identification (mode "indent") or clustering (mode "cluster").
#   4) Fit face identification (mode "indent") or clustering (mode "cluster") and save trained models.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # The training mode ("ident" to train face identification, "cluster" for face clustering)
    parser.add_argument('--mode', type=str, default="ident")
    
    # The video capture input. In case of "None" the default video capture (webcam) is used. Use a filename(s) to read
    # video data from image file (see VideoCapture documentation)
    parser.add_argument('--video', type=str, default="datasets/training_data/Peter_Gilmour/%04d.jpg")
    # parser.add_argument('--video', type=str, default=None)
    
    # Identity label (only required for face identification)
    parser.add_argument('--label', type=str, default="Peter_Gilmour")
    
    args = parser.parse_args()

    if args.video is None:
        camera = cv2.VideoCapture(0)
        wait_for_frame = 1
    else:
        camera = cv2.VideoCapture(args.video)
        wait_for_frame = 100
    camera.set(3, 640)
    camera.set(4, 480)
    
    template = None
    last_position = None
    detector = FaceDetector()

    while True:
        
        char = cv2.waitKey(wait_for_frame) & 0xFF
        if char == 27:
            # Stop capturing using ESC.
            break
        
        ret, frame = camera.read()
        
        if frame is None:
            print("End of stream")
            break
        
        height, width, channels = frame.shape
        if width < 640:
            s = 640.0 / width
            frame = cv2.resize(frame, (int(s*width), int(s*height)))
        # Flip frame if it is live video.
        if args.video is None:
            frame = cv2.flip(frame, 1)
        
        face, template = detector.track_face(ret, frame, template)
        
        if face["rect"] is not None:
            face_rect = face["rect"]
            cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[0] + face_rect[2] - 1, face_rect[1] + face_rect[3] - 1), (0, 255, 0), 2)
            
        cv2.imshow("Camera", frame)
    
    camera.release()
    cv2.destroyAllWindows()
