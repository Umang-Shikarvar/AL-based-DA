import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import time

# Image transform
transform=transforms.Compose([
    transforms.ToTensor()])

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class TIFRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        # Include all .tif images regardless of label presence
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        boxes = []
        labels = []

        # If label file exists, read boxes and labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            cls, x1, y1, x2, y2 = map(float, line.strip().split())
                            boxes.append([x1, y1, x2, y2])
                            labels.append(int(cls) + 1)  # background=0
                        except:
                            pass

        # Convert to tensors, or empty tensors if no labels
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))

# Load dataset
dataset = TIFRCNNDataset(
    image_dir='/home/umang.shikarvar/CycleGAN/lucknow_val/images',
    label_dir='/home/umang.shikarvar/CycleGAN/lucknow_val/labels',
    transforms=transform
)

dataloader = DataLoader(dataset, batch_size=50,pin_memory=True, shuffle=True, collate_fn=collate_fn)

# Load model
backbone = resnet_fpn_backbone(
    backbone_name='resnet50',
    weights=ResNet50_Weights.IMAGENET1K_V1  # Or ResNet50_Weights.DEFAULT
)
# Your number of classes (3 foreground + 1 background)
model = FasterRCNN(backbone, num_classes=4)
model.load_state_dict(torch.load("/home/umang.shikarvar/CycleGAN/delhi_rcnn.pth",map_location=device))
model.to(device)
model.train()

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    start_time = time.time()
    
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass and loss computation
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
    
    end_time = time.time()
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Time: {end_time - start_time:.2f}s")
    torch.save(model.state_dict(), f"/home/umang.shikarvar/CycleGAN/finetuned_source_detectors/model_epoch_{epoch+1}.pth")