from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import torch
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

# Intialize MTCNN for face detection
print("Initializing MTCNN....")
mtcnn = MTCNN(
    image_size=224, thresholds=[0.4, 0.5, 0.5], min_face_size=60, device=device
)

if len(sys.argv) < 2:
    image_path = "./1.jpg"
else:
    image_path = sys.argv[1]

# Load an image containing faces
print("Loading image...")
image = Image.open(image_path)

# Detect faces in the image
print("Detecting...")
boxes, _ = mtcnn.detect(image)

if boxes is not None:
    print(f"Faces found: {len(boxes)}")
    for box in boxes:
        # Draw bounding boxe on image
        draw = ImageDraw.Draw(image)
        draw.rectangle(box.tolist(), outline="red", width=3)
else:
    print("No faces Found")
    exit(0)

print("Showing image...")
image.show()
