import cv2 as cv
import math
import torch
from facenet_pytorch import MTCNN

image_path = None

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
model = 'onnx_model.onnx'
model = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')
     


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
emotion_dict = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear'
}

ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

if torch.cuda.is_available():
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

mtcnn = MTCNN(
    image_size=224, thresholds=[0.4, 0.5, 0.5], min_face_size=60, device=device
)


vid = cv.VideoCapture(image_path if image_path else 0)
padding = 20
if vid is None:
    vid = cv.VideoCapture(1)
if vid is None:
    exit(0)

while vid.grab():
    _, img = vid.retrieve()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.flip(img, 1)

    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
            box = list(map(int, box))
            face = img[max(0, box[1]-padding):min(box[3]+padding,
                                                  img.shape[0]-1), max(0, box[0]-padding):min(box[2]+padding, img.shape[1]-1)]

            blob = cv.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds.argmax()]
            # print(f"Gender: {gender}")

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds.argmax()]
            # print(f"Gender: {age}")

            x1, y1, x2, y2 = box
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 225, 0), 2)

            cv.putText(img, f"{gender},{age}", (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow("window", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break


vid.release()
cv.destroyAllWindows()
