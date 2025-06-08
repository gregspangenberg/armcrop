from armcrop.detect import Detector


class Orient:
    def __init__(self, detector: Detector):
        self.detector = detector


class Crop:
    def __init__(self, detector: Detector):
        self.detector = detector


class Centroids:
    def __init__(self, detector: Detector):
        self.detector = detector
