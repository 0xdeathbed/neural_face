from facenet_pytorch import InceptionResnetV1
from detector import FaceDetector
from torchvision import datasets
import torch
from torch.utils.data import DataLoader


class FaceRecognizer():
    def __init__(self, model="facenet", device="cpu"):
        self.device = device
        self.model = model
        self.recognizer = InceptionResnetV1(
            pretrained='vggface2', device=device).eval()
        self.detector_model = 'mtcnn'
        self.detector = FaceDetector(model=self.detector_model, device=device)

    def register(self, source_dir="saved"):
        detector = FaceDetector(
            model="mtcnn_single_face", device=self.device)

        dataset = datasets.ImageFolder(source_dir)

        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        # print(idx_to_class)

        def collate_fn(batch):
            # if self.device == 'cpu':
            #     return batch[:10]

            return batch[0]

        loader = DataLoader(dataset, collate_fn=collate_fn)

        images_with_classes = {k: [] for k, v in idx_to_class.items()}
        for img, label in loader:
            images_with_classes[label].append(img)

        name_list = []
        embedding_list = []
        for indx, imgs in images_with_classes.items():
            embeddings = []
            for img in imgs:
                face, prob = detector.detector(img, return_prob=True)

                if face is not None and prob > 0.9:
                    emb = self.recognizer(face.unsqueeze(0).to(self.device))
                    embeddings.append(emb.detach())

            if embeddings:
                embeddings = torch.stack(embeddings)
                average_embedings = torch.mean(embeddings, dim=0)

                embedding_list.append(average_embedings)
                name_list.append(idx_to_class[indx])

        data = [embedding_list, name_list]
        torch.save(data, f'{source_dir}/data.pt')

    def recognize(self, image):
        load_data = torch.load('saved/data.pt')
        embedding_list = load_data[0]
        name_list = load_data[1]

        details = []

        img_croppeed_list, prob_list = self.detector.detector(
            image, return_prob=True)

        if img_croppeed_list is not None:
            boxes = self.detector.detect_face(image, use_prob=False)

            for i, prob in enumerate(prob_list):
                if prob > 0.90:
                    emb = self.recognizer(
                        img_croppeed_list[i].unsqueeze(0).to(self.device)).detach()

                    dist_list = []

                    for index, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list)
                    min_dist_idx = dist_list.index(min_dist)

                    name = name_list[min_dist_idx]

                    x1, y1, x2, y2 = boxes[i]
                    if min_dist > 0.90:
                        name = "Unknown"

                    details.append([name, min_dist, boxes[i]])

        return details
