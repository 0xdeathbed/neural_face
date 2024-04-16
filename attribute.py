import cv2 as cv
import torch


class FaceAttribute():
    __ageProto = "model/age_deploy.prototxt"
    __ageModel = "model/age_net.caffemodel"
    __genderProto = "model/gender_deploy.prototxt"
    __genderModel = "model/gender_net.caffemodel"

    __MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    __ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                 '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    __genderList = ['Male', 'Female']

    __padding = 20

    def __init__(self):
        self.age_detector = cv.dnn.readNet(self.__ageModel, self.__ageProto)
        self.gender_detector = cv.dnn.readNet(
            self.__genderModel, self.__genderProto)

        if torch.cuda.is_available():
            self.age_detector.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.age_detector.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
            self.gender_detector.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.gender_detector.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    def detect_age_gender(self, image, box):
        age = self.detect_age(image, box)
        gender = self.detect_gender(image, box)

        return age, gender

    def detect_age(self, image, box):

        face = image[
            max(0, box[1] - self.__padding):min(box[3]+self.__padding, image.shape[0] - 1),
            max(0, box[0] - self.__padding):min(box[2] + self.__padding, image.shape[1]-1)
        ]

        blob = cv.dnn.blobFromImage(
            face, 1.0, (227, 227), self.__MODEL_MEAN_VALUES, swapRB=False)

        self.age_detector.setInput(blob)
        age_preds = self.age_detector.forward()

        age = self.__ageList[age_preds.argmax()]

        return age

    def detect_gender(self, image, box):

        face = image[
            max(0, box[1] - self.__padding):min(box[3]+self.__padding, image.shape[0] - 1),
            max(0, box[0] - self.__padding):min(box[2] + self.__padding, image.shape[1]-1)
        ]

        blob = cv.dnn.blobFromImage(
            face, 1.0, (227, 227), self.__MODEL_MEAN_VALUES, swapRB=False)

        self.gender_detector.setInput(blob)
        gender_preds = self.gender_detector.forward()

        gender = self.__genderList[gender_preds.argmax()]

        return gender
