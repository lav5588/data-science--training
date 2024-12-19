import cv2
import numpy as np

# Load pretrained model files
face_model = "opencv_face_detector_uint8.pb"
face_proto = "opencv_face_detector.pbtxt"
age_model = "age_net.caffemodel"
age_proto = "age_deploy.prototxt"

# Age categories
age_groups = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', 
              '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load models
face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)

# Function to detect faces
def detect_faces(net, frame, conf_threshold=0.7):
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append((x1, y1, x2, y2))
    return bboxes

# Process video
video = cv2.VideoCapture(0)  # Use camera (or replace with video file path)

while True:
    ret, frame = video.read()
    if not ret:
        break

    faces = detect_faces(face_net, frame)
    for (x1, y1, x2, y2) in faces:
        face = frame[y1:y2, x1:x2]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.426337, 87.768914, 114.895847], swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_group = age_groups[age_preds[0].argmax()]
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, age_group, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Age Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
