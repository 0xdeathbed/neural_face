from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn0 = MTCNN(image_size=224, min_face_size=40, device=device)
mtcnn = MTCNN(image_size=224, keep_all=True, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()


if len(sys.argv) > 1 and sys.argv[1] == "register":
    print("Registering faces")
    dataset = datasets.ImageFolder('saved')
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    def collate_fn(batch):
        print(batch)
        return batch[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = []
    embedding_list = []

    for img, idx in loader:
        face, prob = mtcnn0(img, return_prob=True)
        print(face.shape)

        if face is not None and prob > 0.9:

            emb = resnet(face.unsqueeze(0).to(device))
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])

    data = [embedding_list, name_list]
    torch.save(data, 'saved/data.pt')
    exit(0)

load_data = torch.load('saved/data.pt')
embedding_list = load_data[0]
name_list = load_data[1]

cam = cv2.VideoCapture(0)

while cam.grab():
    _, frame = cam.retrieve()

    frame = cv2.flip(frame, 1)

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(
                    0).to(device)).detach()

                dist_list = []

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)
                min_dist_idx = dist_list.index(min_dist)

                name = name_list[min_dist_idx]

                x1, y1, x2, y2 = map(int, boxes[i])
                original_frame = frame.copy()

                if min_dist > 0.90:
                    name = "Unknown"

                cv2.putText(frame,
                            f"{name} {min_dist}", (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("__", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cam.release()
        cv2.destroyAllWindows()
        break
