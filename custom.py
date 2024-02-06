import cv2
import requests
from datetime import datetime 
import time 
# Specify the Plate Recognizer API endpoint and your license key
api_url = "https://api.platerecognizer.com/v1/plate-reader/"
api_key = "357ebf90c0cc6e6ce54510456d9efcb32a770d31"

# Open the camera (0 is typically the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Send the frame to the Plate Recognizer API
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {"upload": ("frame.jpg", img_encoded.tostring(), "image/jpeg")}
    headers = {"Authorization": f"Token {api_key}"}
    response = requests.post(api_url, files=files, headers=headers)

    # Extract license plate information from the response
    result = response.json()
    for plate in result["results"]:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Plate: {plate['plate']} - Confidence: {plate['score']} - DateTime: {formatted_datetime}")
        time.sleep(5)

    # Display the frame with recognized information
    cv2.imshow('ANPR', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
