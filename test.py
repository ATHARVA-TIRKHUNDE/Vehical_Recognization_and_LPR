from util import read_license_plate
from ultralytics import YOLO
import cv2
import time
from datetime import datetime 
from util import key, url
import requests 


YOLO.verbose = False
# Load pre-trained Models
vehical_model = YOLO("yolov8n.pt", verbose=False)
# license_plate_model = YOLO("license_plate_detector.pt", verbose=False)
# license_plate_recognition_model = YOLO("best.pt", verbose=False)

# Open camera
cam = cv2.VideoCapture(0)
# read frame
frame_num = 2
ret = True
vehicles = [2, 3, 5, 7]

while ret:
    frame_num += 1
    ret, frame = cam.read()
    if ret and frame_num%30==0 :
        
        # Vehicle detection
        framec = frame
        detections_generator = vehical_model(framec, verbose=False)[0]
        for detection in detections_generator.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if ( int(class_id) in vehicles ) :
                print("Capture")
                # license_plate_detections = license_plate_model(frame)[0]
                # Send the frame to the Plate Recognizer API
                _, img_encoded = cv2.imencode('.jpg', framec)
                files = {"upload": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
                headers = {"Authorization": f"Token {key}"}
                response = requests.post(url, files=files, headers=headers)

                # Extract license plate information from the response
                result = response.json()
                for plate in result["results"]:
                    current_datetime = datetime.now()
                    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    platetext = plate['plate'].upper()
                    print(f"Plate: {platetext} - Confidence: {plate['score']} - DateTime: {formatted_datetime}")
                print("Done")


# Release the camera
cam.release()
cv2.destroyAllWindows()
