import numpy as np
from retinaface.pre_trained_models import get_model
from PIL import Image, ImageDraw
import torch
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

# Intialize MTCNN for face detection
print("Initializing Retinaface....")
model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
model.eval()

if len(sys.argv) < 2:
    image_path = "./1.jpg"
else:
    image_path = sys.argv[1]

# Load an image containing faces
print("Loading image...")
image = Image.open(image_path)

# Detect faces in the image
print("Detecting...")
boxes = model.predict_jsons(np.array(image), confidence_threshold=0.8)

if boxes is not None:
    print(f"Faces found: {len(boxes)}")
    for box in boxes:
        box = box["bbox"]
        # Draw bounding boxe on image
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=3)
else:
    print("No faces Found")
    exit(0)

print("Showing image...")
image.show()
