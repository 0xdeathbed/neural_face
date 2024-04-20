import argparse
import torch
from detector import FaceDetector
from recognizer import FaceRecognizer
from PIL import ImageDraw, Image
import os
import shutil
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Detect and Recognize faces in image")
parser.add_argument("--register", action="store_true",
                    help="Register this image")
parser.add_argument("--cpu", action="store_true", help="Explicitly use cpu")
parser.add_argument("-f", action="store", help="Path to an image")
parser.add_argument("--detect", action="store_true",
                    help="Detect face in image")
parser.add_argument("--save", action="store_true",
                    help="Save detected or recognized image in current directory")
parser.add_argument("--recognize", action="store_true",
                    help="Recognize unknown face from registered faces")

args = parser.parse_args()


def detection(file):
    face_detector = FaceDetector(model="mtcnn")

    print("Loading Image..")
    image = Image.open(file)

    print("Detecting faces....")
    boxes = face_detector.detect_face(image)

    if boxes is not None:
        print(f"Probable Face found: {len(boxes)}")
        for box in boxes:
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, outline="red", width=4)

        if args.save:
            save_img = f"detected_{file}"
            print(f"Saved detected image as: {save_img}")
            image.save(save_img)
        else:
            print("Showing detected image")
            image.show()
    else:
        print(f"No faces found in image: {file} ")
        exit(0)


def register_face(file):
    recognizer = FaceRecognizer(device=device)
    name = input("Enter the name under which image to be registered: ")

    print(f"Registering {name}/{file} image")

    os.mkdir(f"saved/{name}")
    shutil.copy(args.f, f"saved/{name}/{file}")

    recognizer.register()

    exit(0)


def recognize_face(file):
    recognizer = FaceRecognizer(device=device)

    print("Loading Image")
    image = Image.open(file)
    frame = np.array(image)

    details = recognizer.recognize(image)

    for name, min_dist, box in details:

        x1, y1, x2, y2 = box
        text = f"{name},{min_dist:.2f}"

        cv2.putText(frame,
                    text, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    image = Image.fromarray(frame)

    if args.save:
        save_img = f"recognize_{file}"
        print(f"Saved recognized image as: {save_img}")
        image.save(save_img)
    else:
        print("Showing recgonized image")
        image.show()


if __name__ == '__main__':
    if args.cpu:
        device = "cpu"
    print("Running on device: {}".format(device))

    if args.f is not None:
        if args.recognize:
            recognize_face(args.f)

        elif args.detect:
            detection(args.f)

        elif args.register:
            register_face(args.f)
