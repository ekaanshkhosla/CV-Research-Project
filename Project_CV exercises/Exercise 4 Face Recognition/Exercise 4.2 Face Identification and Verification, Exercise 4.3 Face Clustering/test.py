import argparse
import cv2
from face_detector import FaceDetector
from face_recognition import FaceRecognizer
from face_recognition import FaceClustering

# The test module of the face recognition system. This comprises the following workflow:
#   1) Capturing new video frame.
#   2) Run face detection / tracking.
#   3) Extract face embedding and perform face identification (mode "indent") or re-identification (mode "cluster").
#   4) Display face detection / tracking along with the prediction of face identification.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # The training mode ("ident" to train face identification, "cluster" for face clustering)
    parser.add_argument('--mode', type=str, default="ident")
    
    # The video capture input. In case of "None" the default video capture (webcam) is used. Use a filename(s) to read
    # video data from image file (see VideoCapture documentation)
    parser.add_argument('--video', type=str, default="datasets/test_data/Nancy_Sinatra/%04d.jpg")
    
    args = parser.parse_args()

    # Setup OpenCV video capture.
    if args.video is None:
        camera = cv2.VideoCapture(0)
        wait_for_frame = 200
    else:
        camera = cv2.VideoCapture(args.video)
        wait_for_frame = 100
    camera.set(3, 640)
    camera.set(4, 480)

    # Image display
    cv2.namedWindow("Camera")
    cv2.moveWindow("Camera", 0, 0)

    # Prepare face detection, identification, and clustering.
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    clustering = FaceClustering()

    # The video capturing loop.
    on_track = False
    
    max_distance = []
    min_prob = []
    template = None
    
    while True:

        char = cv2.waitKey(wait_for_frame) & 0xFF
        if char == 27:
            # Stop capturing using ESC.
            break

        # Capture new video frame.
        ret, frame = camera.read()
        if frame is None:
            print("End of stream")
            break
        # Resize the frame.
        height, width, channels = frame.shape
        if width < 640:
            s = 640.0 / width
            frame = cv2.resize(frame, (int(s * width), int(s * height)))
        # Flip frame if it is live video.
        if args.video is None:
            frame = cv2.flip(frame, 1)

        # Track (or initially detect if required) a face in the current frame.
        face, template = detector.track_face(ret, frame, template)

        if face is not None and not on_track:
            # We found a new face that we can track over time.
            on_track = True

            if args.mode == "ident":
                # Face identification: predict identity for the current frame.
                predicted_label, prob, dist_to_prediction = recognizer.predict(face["aligned"])
                
                max_distance.append(dist_to_prediction)
                min_prob.append(prob)
                
                label_str = "{}".format(predicted_label)
                confidence_str = "Prob.: {:1.2f}, Dist.: {:1.2f}".format(prob, dist_to_prediction)
                
                if(dist_to_prediction > 1.0 or prob < 0.95):
                    label_str = "Unknown"
                
                
            if args.mode == "cluster":
                # Face clustering: determine cluster for the current frame.
                predicted_label, distances_to_clusters = clustering.predict(face["aligned"])
                formatted_distances = ["{:1.2f}".format(dist) for dist in distances_to_clusters]
                label_str = "Cluster {}".format(predicted_label)
                confidence_str = "Dist.: {}".format(formatted_distances)


        if face is None or face["response"] < detector.tm_threshold:
            # We lost the track of the face visible in the previous frame.
            on_track = False


        # Display annotations for face tracking, identification, and clustering.
        if face is not None:
            face_rect = face["rect"]
            cv2.rectangle(frame, (face_rect[0], face_rect[1]),(face_rect[0] + face_rect[2] - 1, face_rect[1] + face_rect[3] - 1),(0, 255, 0), 2)
            cv2.putText(frame, label_str, (face_rect[0] + face_rect[2] + 10, face_rect[1] + face_rect[3] + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, confidence_str, (face_rect[0] + face_rect[2] - 100, face_rect[1] + face_rect[3] + 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Camera", frame)
    

    camera.release()
    cv2.destroyAllWindows()