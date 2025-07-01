import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList
from tqdm import tqdm

# Set device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Load model
backbone = resnet_fpn_backbone(
    backbone_name='resnet50',
    weights=ResNet50_Weights.IMAGENET1K_V1
)
model = FasterRCNN(backbone, num_classes=4)
model.load_state_dict(torch.load("/home/umang.shikarvar/CycleGAN/gen_delhi_rcnn.pth", map_location=device))
model.roi_heads.nms_thresh = 0.33
model.to(device)
model.eval()

# Image paths
image_dir = "/home/umang.shikarvar/CycleGAN/lucknow_airshed/val/images"
transform = transforms.Compose([transforms.ToTensor()])
image_uncertainties = []

def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy

# Loop over images
with torch.no_grad():
    for filename in tqdm(os.listdir(image_dir)):
        if not filename.lower().endswith((".jpg", ".png", ".tif")):
            continue

        img_path = os.path.join(image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        image_size = [tuple(image_tensor.shape[-2:])]  # [(H, W)]

        # Backbone feature extraction
        features = model.backbone(image_tensor)

        # Wrap image in ImageList
        images = ImageList(image_tensor, image_size)

        # RPN proposals
        proposals, _ = model.rpn(images, features)

        if len(proposals[0]) == 0:
            image_uncertainties.append((filename, 0.0))
            continue

        # ROI head class logits
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_size)
        box_features = model.roi_heads.box_head(box_features)
        class_logits = model.roi_heads.box_predictor.cls_score(box_features)

        entropy = compute_entropy(class_logits)
        avg_entropy = entropy.mean().item()
        image_uncertainties.append((filename, avg_entropy))

# Sort and save top-K uncertain images
image_uncertainties.sort(key=lambda x: x[1], reverse=True)
top_k = 10
with open("/home/umang.shikarvar/CycleGAN/gen_source_top_uncertain_images.txt", "w") as f:
    for fname, ent in image_uncertainties[:top_k]:
        f.write(f"{fname}\n")

print(f"Saved top {top_k} uncertain image filenames to top_uncertain_images.txt")