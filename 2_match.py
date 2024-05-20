from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import NewType, Optional
import numpy as np
from numpy.typing import NDArray
from rtree import index

import msgspec
from data import Detection, FrameDetections
import cv2


video_path = "./videos/_stream_03.mp4"
cap = cv2.VideoCapture(video_path)
json_path = Path(video_path).with_suffix(".json")
json_file = iter(open(json_path))
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


@dataclass(slots=True, frozen=True)
class Box:
    x1: float
    y1: float
    y2: float
    x2: float

    def intersects(self, other: Box) -> bool:
        return not (
            self.x2 < other.x1
            or self.x1 > other.x2
            or self.y2 < other.y1
            or self.y1 > other.y2
        )

    def intersection(self, other: Box) -> Optional[Box]:
        if not self.intersects(other):
            return None
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            return Box(x1=inter_x1, y1=inter_y1, x2=inter_x2, y2=inter_y2)
        return None

    def union(self, other: Box) -> Box:
        uni_x1 = min(self.x1, other.x1)
        uni_y1 = min(self.y1, other.y1)
        uni_x2 = max(self.x2, other.x2)
        uni_y2 = max(self.y2, other.y2)
        return Box(x1=uni_x1, y1=uni_y1, x2=uni_x2, y2=uni_y2)

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: Box) -> float:
        intersection = self.intersection(other)
        if intersection is None:
            return 0
        union = self.union(other)
        return intersection.area() / union.area()

    def astuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


BoxID = NewType("BoxID", int)
PatternID = NewType("PatternID", int)
Timestamp = NewType("Timestamp", float)
Seconds = NewType("Seconds", float)


def keypoints_within_box(
    box: tuple[float, ...],
    keypoints_in: NDArray,
    descriptors_in: NDArray,
    edgeThreshold: float = 5,
):
    x_min = box[0] + edgeThreshold
    y_min = box[1] + edgeThreshold
    x_max = box[2] - edgeThreshold
    y_max = box[3] - edgeThreshold
    x = keypoints_in[:, 0]
    y = keypoints_in[:, 1]
    mask = (y_min <= y) & (y <= y_max) & (x_min <= x) & (x <= x_max)
    return keypoints_in[mask], descriptors_in[mask]


@dataclass
class Pattern:
    box: Box
    keypoints: NDArray
    descriptors: NDArray

    @staticmethod
    def from_detection(detection: Detection) -> Pattern:
        box = Box(x1=detection.x1, y1=detection.y1, x2=detection.x2, y2=detection.y2)
        return Pattern(
            box=box,
            keypoints=np.array(detection.keypoints),
            descriptors=np.array(detection.descriptors, dtype=np.uint8),
        )

    def crop(self, box: Box) -> Pattern:
        new_keypoints, new_descriptors = keypoints_within_box(
            box.astuple(), self.keypoints, self.descriptors
        )
        return Pattern(box, new_keypoints, new_descriptors)

    def similarity(self, other: Pattern) -> float:
        intersection = self.box.intersection(other.box)
        if not intersection:
            return 0
        return self.crop(intersection)._similarity_unchecked(other.crop(intersection))

    def _similarity_unchecked(self, other: Pattern) -> float:
        # shitcode, you can do withoud bf matcher
        matches = matcher.match(self.descriptors, other.descriptors)
        if not matches:
            return 0
        num_matched = len(matches)

        pct_matched = num_matched / min(len(self.descriptors), len(other.descriptors))
        if pct_matched < 0.8:  # fix this hardcode!
            return 0
        dist_thresh = 10 # Hardcode
        idx_self = [i.queryIdx for i in matches]
        idx_other = [i.trainIdx for i in matches]

        pts_self = self.keypoints[idx_self]
        pts_other = other.keypoints[idx_other]
        dist = np.linalg.norm(pts_self - pts_other, axis=1) # fixme, use squared euclidean for speed gains in real code!
        pct_inlier = sum(dist < dist_thresh) / len(dist)
        print(pct_inlier)

        return pct_inlier


class SusBoxDetector:
    def __init__(self) -> None:
        self.box_index = 0
        self.rtree = index.Index(interleaved=True)
        self.box_dict: dict[BoxID, Box] = {}
        self.box_appeared: dict[BoxID, Timestamp] = {}
        self.box_timeout = Seconds(60)

    def find_sus(
        self,
        detections: list[Detection],
        iou_thresh=0.95,
        num_to_sus=10,
    ) -> list[bool]:
        sus: list[bool] = []
        for i in detections:
            box = Box(x1=i.x1, y1=i.y1, x2=i.x2, y2=i.y2)
            box_tup = box.astuple()
            intersection_ids = self.rtree.intersection(box_tup, objects=False)
            intersections = [self.box_dict[BoxID(j)] for j in intersection_ids]
            ious = [box.iou(j) for j in intersections]
            high_ious = [j for j in ious if j > iou_thresh]
            is_sus = len(high_ious) >= num_to_sus
            sus.append(is_sus)
        return sus

    def update(self, detections: list[Detection], timestamp: Timestamp) -> None:
        for i in detections:
            box = Box(x1=i.x1, y1=i.y1, x2=i.x2, y2=i.y2)
            self.rtree.insert(self.box_index, box.astuple())
            box_id = BoxID(self.box_index)
            self.box_appeared[box_id] = timestamp
            self.box_dict[box_id] = box
            self.box_index += 1

        for old_box_id in self.get_old_boxes_id(timestamp):
            self.rtree.delete(old_box_id, self.box_dict[old_box_id].astuple())
            del self.box_dict[old_box_id]
            del self.box_appeared[old_box_id]

    def get_old_boxes_id(self, timestamp: Timestamp) -> list[BoxID]:
        old_boxes = []
        for box, appered in self.box_appeared.items():
            diff = timestamp - appered
            if diff > self.box_timeout:
                old_boxes.append(box)
        return old_boxes


class PatternDetector:
    def __init__(self) -> None:
        self.pattern_ids = 0
        self.pattern_dict: dict[PatternID, Pattern] = {}
        self.pattern_sightings: dict[PatternID, Timestamp] = {}
        self.pattern_timeout = Seconds(60)

    def detect(self, sus_detections: list[Detection]) -> list[Optional[PatternID]]:
        patterns = []
        for i in sus_detections:
            patterns.append(self._try_find_pattern(i))
        return patterns

    def _try_find_pattern(
        self, detection: Detection, iou_thresh: float = 0.8, sim_thresh: float = 0.9
    ) -> Optional[PatternID]:
        query = Pattern.from_detection(detection)

        best_sim = 0.0
        best_pattern_id = None
        for pattern_id, pattern in self.pattern_dict.items():
            iou = pattern.box.iou(query.box)
            if iou < iou_thresh:
                continue
            similarity = query.similarity(pattern)
            if similarity < sim_thresh:
                continue

            if similarity > best_sim:
                best_sim = similarity
                best_pattern_id = pattern_id

        return best_pattern_id

    def update_seen(
        self, seen_pattern_ids: list[PatternID], timestamp: Timestamp
    ) -> None:
        for i in seen_pattern_ids:
            self.pattern_sightings[i] = timestamp
        for i in self.get_old_patterns_id(timestamp):
            del self.pattern_dict[i]
            del self.pattern_sightings[i]

    def add_new_patterns(
        self, detections: list[Detection], timestamp: Timestamp
    ) -> None:
        for i in detections:
            new_id = PatternID(self.pattern_ids)
            self.pattern_ids += 1
            self.pattern_dict[new_id] = Pattern.from_detection(i)
            self.pattern_sightings[new_id] = timestamp

    def get_old_patterns_id(self, timestamp: Timestamp) -> list[PatternID]:
        old_patterns = []
        for ph_id, appered in self.pattern_sightings.items():
            diff = timestamp - appered
            if diff > self.pattern_timeout:
                old_patterns.append(ph_id)
        return old_patterns

    def detect_update(
        self, sus_detections: list[Detection], timestamp: Timestamp
    ) -> list[bool]:
        pattern_ids = self.detect(sus_detections)
        seen_ids = [i for i in pattern_ids if i is not None]
        is_pattern = [i is not None for i in pattern_ids]
        self.update_seen(seen_ids, timestamp)

        non_pattern_detections = []
        for id, det in zip(pattern_ids, detections):
            if id is None:
                non_pattern_detections.append(det)
        self.add_new_patterns(non_pattern_detections, timestamp)

        return is_pattern


sus_det = SusBoxDetector()
phantom_det = PatternDetector()

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    next_json = next(json_file)
    frame_info = msgspec.json.decode(next_json, type=FrameDetections)  # shit name!
    detections = frame_info.detections
    timestamp = Timestamp(frame_id * 1 / 6)

    is_sus = sus_det.find_sus(detections)

    sus_detections = [detections[i] for i in range(len(detections)) if is_sus[i]]
    sus_index = [i for i in range(len(detections)) if is_sus[i]]

    is_phantom = phantom_det.detect_update(sus_detections, timestamp)
    sus_det.update(detections, timestamp)

    sus_count = 0
    for box_id in range(len(detections)):
        box = detections[box_id]
        box_sus = is_sus[box_id]

        box_phantom = False
        if box_sus:
            box_phantom = is_phantom[sus_count]
            sus_count += 1
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0) if not box_phantom else (0, 0, 255),
            2,
        )

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.1)

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
