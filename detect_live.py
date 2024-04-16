import torch
import cv2 as cv
from detector import FaceDetector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

detector = FaceDetector(model="mtcnn")

vid = cv.VideoCapture(0)
if vid is None:
    vid = cv.VideoCapture(1)
if vid is None:
    exit(0)

while vid.grab():
    _, img = vid.retrieve()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.flip(img, 1)

    boxes = detector.detect_face(img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 225, 0), 2)

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow("window", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break


vid.release()
cv.destroyAllWindows()
