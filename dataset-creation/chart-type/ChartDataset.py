from torch.utils.data import Dataset
from PIL import Image
import os

class ChartDataset(Dataset):
    def __init__(self, img_dir, id_label_map, class_to_idx, transform=None):
        self.img_dir = img_dir
        self.id_label_map = id_label_map
        self.transform = transform
        self.classes = sorted(set(id_label_map.values()))
        # self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # This is coming from training dataset only
        self.class_to_idx = class_to_idx
        self.ids = list(id_label_map.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[self.id_label_map[img_id]]
        if self.transform:
            image = self.transform(image)
        return image, label