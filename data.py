import msgspec


class Detection(msgspec.Struct):
    x1: int
    x2: int
    y1: int
    y2: int
    keypoints: list[tuple[float, float]]
    descriptors: list[list[int]]


class FrameDetections(msgspec.Struct):
    detections: list[Detection]
    frame_idx: int
