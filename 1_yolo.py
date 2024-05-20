from pathlib import Path
import time
import cv2
from ultralytics import YOLO
import numpy as np
from numpy.typing import NDArray
import msgspec

from data import Detection, FrameDetections


# Load the YOLOv8 model
model = YOLO("./weights/det-seg-yolov8.pt")

# Open the video file
video_path = "./videos/_stream_03.mp4"

cap = cv2.VideoCapture(video_path)
orb = cv2.ORB_create(
    nfeatures=10,
    scaleFactor=1.2,
    nlevels=4,
    edgeThreshold=5,
    firstLevel=0,
    WTA_K=2,
    patchSize=10,
    fastThreshold=7,
)

def offset_keypoints(
    keypoints: list[cv2.KeyPoint], offset_x: int, offset_y: int
) -> list[cv2.KeyPoint]:
    adjusted_keypoints = []
    for kp in keypoints:
        x, y = kp.pt
        adjusted_keypoints.append(
            cv2.KeyPoint(
                x + offset_x,
                y + offset_y,
                kp.size,
                kp.angle,
                kp.response,
                kp.octave,
                kp.class_id,
            )
        )
    return adjusted_keypoints


def align_to_multiple_of(coord, box_size):
    return (coord // box_size) * box_size


def divide_into_smaller_squares(x1, y1, x2, y2, box_size=50):
    x1 = align_to_multiple_of(x1, box_size)
    y1 = align_to_multiple_of(y1, box_size)
    x2 = align_to_multiple_of(x2 + box_size, box_size)
    y2 = align_to_multiple_of(y2 + box_size, box_size)

    squares = []

    for x in range(x1, x2, box_size):
        for y in range(y1, y2, box_size):
            squares.append((x, y, x + box_size, y + box_size))
    return squares


def keypoints_in_boxes(gray_frame: NDArray, box: tuple[int, ...]) -> tuple[list[cv2.KeyPoint], list[NDArray[np.uint8]]]:
    keypoints = []
    descriptors = []
    for x1, y1, x2, y2 in divide_into_smaller_squares(
        box[0], box[1], box[2], box[3], box_size=50
    ):
        b = gray_frame[y1:y2, x1:x2]
        kp, des = orb.detectAndCompute(b, None)
        if not kp:
            continue
        kp = offset_keypoints(kp, x1, y1)
        keypoints.extend(kp)
        descriptors.extend(des)
    return keypoints_within_box(box, keypoints, descriptors)

def keypoints_within_box(box: tuple[int, ...], keypoints_in: list[cv2.KeyPoint], descriptors_in: list[NDArray]):
    keypoints: list[cv2.KeyPoint] = []
    descriptors: list[NDArray] = []
    x_min = box[0]
    y_min = box[1]
    x_max = box[2]
    y_max = box[3]
    
    for kp, des in zip(keypoints_in, descriptors_in):
        x, y = kp.pt
        if (y_min <= y <= y_max) and (x_min <= x <= x_max):
            keypoints.append(kp)
            descriptors.append(des)
    return keypoints, descriptors

json_path = Path(video_path).with_suffix(".json")
frame_id = 0

with open(json_path, "wb") as json_file:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = []

        # Draw the results on the frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                cls_name = model.names[int(cls)]
                if cls_name != "person":
                    continue
                
                keypoints, descriptors = keypoints_in_boxes(grayscale_frame, (x1, y1, x2, y2))
                frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put the label
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    keypoints=[(i.pt[0], i.pt[1]) for i in keypoints],
                    descriptors=[i.tolist() for i in descriptors]
                ))
                
        frame_detections = FrameDetections(
            detections, 
            frame_id
        )
        
        json_file.write(msgspec.json.encode(frame_detections))
        json_file.write("\n".encode())
        
        frame_id += 1

# Release everything
cap.release()
cv2.destroyAllWindows()
