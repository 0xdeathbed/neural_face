from facenet_pytorch import MTCNN
from retinaface.pre_trained_models import get_model
import numpy as np


class FaceDetector():

    def __init__(self, model='retinaface', device='cpu'):
        self.model = model
        if self.model == 'retinaface':
            self.detector = get_model("resnet50_2020-07-20",
                                      max_size=2048, device=device)
            self.detector.eval()
        elif self.model == 'mtcnn':
            self.detector = MTCNN(
                image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60, device=device
            )
        elif self.model == 'mtcnn_single_face':
            self.detector = MTCNN(
                image_size=224, min_face_size=40, device=device
            )
        else:
            raise ValueError("Model value is not correct")

    def cropped_image(self, box, image):
        x_min, y_min, x_max, y_max = box

        x_min = np.clip(x_min, 0, x_max)
        y_min = np.clip(y_min, 0, y_max)

        img = image[y_min:y_max, x_min:x_max]

        return img

    def detect_face(self, image, use_prob=True):
        if self.model == 'retinaface':
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            annotations = self.detector.predict_jsons(
                image, confidence_threshold=0.8)

            boxes = None
            if annotations is not None:
                boxes = [annotation["bbox"] for annotation in annotations]

        else:
            faces_box, probs = self.detector.detect(image)
            boxes = []
            if faces_box is not None:
                for box, prob in zip(faces_box, probs):
                    if use_prob:
                        if prob > 0.90:
                            box = list(map(int, box))
                            boxes.append(box)
                    else:
                        box = list(map(int, box))
                        boxes.append(box)

            if not boxes:
                boxes = None

        return boxes
