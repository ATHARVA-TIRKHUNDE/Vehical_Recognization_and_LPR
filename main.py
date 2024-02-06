from util import read_license_plate
from ultralytics import YOLO
import cv2
import time
YOLO.verbose = False
# Load pre-trained Models
vehical_model = YOLO("yolov8n.pt")
license_plate_model = YOLO("license_plate_detector.pt")
license_plate_recognition_model = YOLO("best.pt")

# Open camera
cam = cv2.VideoCapture(0)

# read frame
frame_num = -1
ret = True
vehicles = [2, 3, 5, 7]

while ret:
    frame_num += 1
    ret, frame = cam.read()
    
    if ret :
        # Vehicle detection
        detections_generator = vehical_model(frame)[0]
        
        for detection in detections_generator.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            if int(class_id) in vehicles:
                # License plate detection
                license_plate_detections = license_plate_model(frame)[0]
                
                for license_plate in license_plate_detections.boxes.data.tolist():
                    x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id_lp = license_plate
                    
                    # Crop license plate
                    license_plate_crop = frame[int(y1_lp):int(y2_lp), int(x1_lp): int(x2_lp), :]

                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # License plate recognition
                    license_plate_result = license_plate_recognition_model(license_plate_crop_thresh)[0]

                    # Extract license plate text from the result
                    plate_text = read_license_plate(license_plate_result)

                    if plate_text is not None:
                        print("License Plate Text:", plate_text)

                    time.sleep(5)  # Sleep for 5 seconds

# Release the camera
cam.release()
cv2.destroyAllWindows()
