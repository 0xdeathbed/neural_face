from PIL import Image, ImageDraw
from detector import FaceDetector
import torch
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

face_detector = FaceDetector(model='mtcnn')

if len(sys.argv) < 2:
    image_path = "./1.jpg"
else:
    image_path = sys.argv[1]

# Load an image containing faces
print("Loading image...")
image = Image.open(image_path)

# Detect faces in the image
print("Detecting...")
boxes = face_detector.detect_face(image)

if boxes is not None:
    print(f"Faces found: {len(boxes)}")
    for box in boxes:
        # Draw bounding boxe on image
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=3)
else:
    print("No faces Found")
    exit(0)

print("Showing image...")
image.show()
