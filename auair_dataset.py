# auair_dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json



class AUAirCocoDataset(Dataset):
    def __init__(self, images_dir, annotation_path, processor, transforms=None):
        # Load COCO-format JSON
        with open(annotation_path) as f:
            coco = json.load(f)

        self.images_dir = images_dir
        self.processor = processor
        self.transforms = transforms

        # Build id-to-image dictionary
        self.image_id_to_info = {img['id']: img for img in coco['images']}

        # Organize annotations by image_id
        self.image_id_to_annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        self.ids = list(self.image_id_to_info.keys())
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.image_id_to_info[img_id]
        file_path = os.path.join(self.images_dir, img_info['file_name'])

        # Load image
        image = Image.open(file_path).convert("RGB")

        # Get annotations
        anns = self.image_id_to_annotations.get(img_id, [])

        boxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]

        # Convert (x, y, w, h) to (x_min, y_min, x_max, y_max) (coordinate format)
        boxes = torch.tensor(boxes, dtype=torch.float)
        boxes[:, 2:] += boxes[:, :2]

        target = {
            "image_id": torch.tensor([img_id]),
            "class_labels": labels,
            "boxes": boxes
        }

        # Apply DETR processor
        encoding = self.processor(
            image,
            annotations={"image_id": img_id, "annotations": anns},
            return_tensors="pt"
        )

        # Only squeeze tensor fields
        encoding = {
            k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in encoding.items()
        }

        return encoding