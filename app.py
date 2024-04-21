import argparse
import torch
from detector import FaceDetector
from recognizer import FaceRecognizer
from attribute import FaceAttribute
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
                    help="Detect face in image [-f should point to valid img]")
parser.add_argument("--save", action="store_true",
                    help="Save detected or recognized image in current directory")
parser.add_argument("--recognize", action="store_true",
                    help="Recognize unknown face from registered faces")
parser.add_argument("--attribute", action="store_true",
                    help="Estimate age and gender of face in image")
parser.add_argument("--live", action="store_true", help="To enable live cam")
parser.add_argument("-l", action="store", default="detect", choices=["detect", "recognize", "attribute"],
                    help="Which feature to use for live cam [to be used with --live arg]")

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


def recognize_face(file, attribute=False):
    recognizer = FaceRecognizer(device=device)
    if attribute:
        attribute_detector = FaceAttribute()

    print("Loading Image")
    image = Image.open(file)
    frame = np.array(image)

    details = recognizer.recognize(image)

    for name, min_dist, box in details:
        attribute_text = ""
        if attribute:
            age, gender = attribute_detector.detect_age_gender(
                image=frame, box=box)

            attribute_text = f"{gender},{age}"

        x1, y1, x2, y2 = box
        text = f"{name},{min_dist:.2f} [{attribute_text}]"

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


def attribute(file):
    detector = FaceDetector(model='mtcnn')
    attribute_detector = FaceAttribute()

    print("Loading Image")
    frame = cv2.imread(file)

    boxes = detector.detect_face(frame)

    if boxes is not None:
        for box in boxes:
            age, gender = attribute_detector.detect_age_gender(
                image=frame, box=box)

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 2)

            cv2.putText(frame, f"{gender},{age}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    image = Image.fromarray(frame)

    if args.save:
        save_img = f"attribute_{file}"
        print(f"Saved detected attribute image as: {save_img}")
        image.save(save_img)
    else:
        print("Showing recgonized image")
        image.show()


def live_detect(video_src=0):
    detector = FaceDetector(model="mtcnn")

    vid = cv2.VideoCapture(video_src)
    if vid is None:
        exit(0)

    while vid.grab():
        _, img = vid.retrieve()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)

        boxes = detector.detect_face(img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 225, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("window", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


def live_recognize(video_src=0):
    recognizer = FaceRecognizer(device=device)

    cam = cv2.VideoCapture(video_src)
    if cam is None:
        exit(0)

    while cam.grab():
        _, frame = cam.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        img = Image.fromarray(frame)

        details = recognizer.recognize(img)

        for name, min_dist, box in details:

            x1, y1, x2, y2 = box

            cv2.putText(frame,
                        f"{name} {min_dist:.3f}", (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("__", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def live_attribute(video_src=0):
    detector = FaceDetector(model='mtcnn')
    attribute_detector = FaceAttribute()

    vid = cv2.VideoCapture(video_src)
    if vid is None:
        exit(0)

    while vid.grab():
        _, img = vid.retrieve()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)

        boxes = detector.detect_face(img)

        if boxes is not None:
            for box in boxes:
                box = list(map(int, box))

                age, gender = attribute_detector.detect_age_gender(
                    image=img, box=box)

                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 225, 0), 2)

                cv2.putText(img, f"{gender},{age}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("window", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if args.cpu:
        device = "cpu"
    print("Running on device: {}".format(device))

    if args.live:
        if args.l == 'detect':
            live_detect()
        elif args.l == 'recognize':
            live_recognize()
        elif args.l == 'attribute':
            live_attribute()
        else:
            print("Invalid value: -l detect|recognize|attribute")
    elif args.f is not None:
        if args.recognize:
            recognize_face(args.f, args.attribute)
        elif args.attribute:
            attribute(args.f)
        elif args.detect:
            detection(args.f)
        elif args.register:
            register_face(args.f)
