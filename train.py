import torch
from itertools import zip_longest
from dataset import SourceDataset, TargetDataset
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms

# Seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 200
BATCH_SIZE = 6
λ_det = 5.0
λ_adv = 0.5
λ_id  = 5.0
λ_cycle  = 10.0

print(f"Using device: {device}")

def collate_fn(batch):
    return tuple(zip(*batch))

transform=transforms.Compose([
    transforms.ToTensor()])

# Load dataset
source = SourceDataset(
    image_dir='/home/umang.shikarvar/Style_GAN/delhi/images',
    label_dir='/home/umang.shikarvar/Style_GAN/delhi/labels',
    transforms=transform
)
source_loader = DataLoader(source, batch_size=BATCH_SIZE,pin_memory=True, num_workers=8, shuffle=True, collate_fn=collate_fn)

target= TargetDataset(
    image_dir='/home/umang.shikarvar/Style_GAN/lucknow/images',
    transforms=transform
)
target_loader = DataLoader(target, batch_size=BATCH_SIZE, pin_memory=True, num_workers=8, shuffle=True)

# Load backbone with pretrained weights
backbone = resnet_fpn_backbone(
    backbone_name='resnet50',
    weights=ResNet50_Weights.IMAGENET1K_V1  # Or ResNet50_Weights.DEFAULT
)
# Your number of classes (3 foreground + 1 background)
detector = FasterRCNN(backbone, num_classes=4)
detector.load_state_dict(torch.load('/home/umang.shikarvar/CycleGAN/delhi_rcnn.pth', map_location=device))
detector.to(device)
detector.train()

# Generators
G_AB = Generator(img_channels=3).to(device)  # Source to Target
G_BA = Generator(img_channels=3).to(device)  # Target to Source

# Discriminators
D_B = Discriminator().to(device)  # Discriminator for target domain
D_A = Discriminator().to(device)  # Discriminator for source domain

# Optimizers
opt_G = Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=1e-4)
opt_D = Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=1e-4)

scheduler_G = CosineAnnealingWarmRestarts(opt_G, T_0=50, T_mult=2)
scheduler_D = CosineAnnealingWarmRestarts(opt_D, T_0=50, T_mult=2)
for param in detector.parameters():
    param.requires_grad = False

L1_loss = nn.L1Loss()

warmup_epochs = 15  # Number of epochs to warm up G and D

for epoch in range(num_epochs):
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    epoch_loss_det = 0.0
    epoch_loss_adv = 0.0
    epoch_loss_id  = 0.0
    total_batches = min(len(source_loader), len(target_loader))
    for batch_index, (batch_s, batch_t) in enumerate(zip_longest(source_loader, target_loader)):
        if batch_s is None or batch_t is None:
            continue

        x_s, y_s = batch_s
        x_t = batch_t

        x_s = torch.stack(x_s).to(device)
        y_s = [{k: v.to(device) for k, v in tgt.items()} for tgt in y_s]
        x_t = x_t.to(device)

        # 1. Translate source → target and back: x_s → x_s2t → x_s2t2s
        x_s2t = G_AB(x_s)
        x_s2t2s = G_BA(x_s2t)

        # 2. Translate target → source and back: x_t → x_t2s → x_t2s2t
        x_t2s = G_BA(x_t)
        x_t2s2t = G_AB(x_t2s)

        # 3. Cycle loss
        loss_cycle = L1_loss(x_s2t2s, x_s) + L1_loss(x_t2s2t, x_t)

        # 4. Identity loss
        x_t_id = G_AB(x_t)
        loss_id = L1_loss(x_t_id, x_t)
        x_s_id = G_BA(x_s)
        loss_id += L1_loss(x_s_id, x_s)

        # 5. Discriminator loss 
        pred_D_B_real = D_B(x_t)
        pred_D_B_fake = D_B(x_s2t.detach())
        pred_D_A_real = D_A(x_s)
        pred_D_A_fake = D_A(x_t2s.detach())

        loss_D_B = mse_loss(pred_D_B_real, torch.ones_like(pred_D_B_real)) + mse_loss(pred_D_B_fake, torch.zeros_like(pred_D_B_fake))

        loss_D_A = mse_loss(pred_D_A_real, torch.ones_like(pred_D_A_real)) + mse_loss(pred_D_A_fake, torch.zeros_like(pred_D_A_fake))

        loss_D = loss_D_B + loss_D_A

        # 6. Generator adversarial loss 
        loss_adv = mse_loss(D_B(x_s2t), torch.ones_like(D_B(x_s2t))) + mse_loss(D_A(x_t2s), torch.ones_like(D_A(x_t2s)))

        # 7. Detector loss (only after warmup)
        if epoch >= warmup_epochs:
            loss_dict = detector(x_s2t, y_s)
            loss_det = sum(loss for loss in loss_dict.values())
        else:
            loss_det = torch.tensor(0.0, device=device)

        # 8. Total generator loss 
        loss_G = λ_det * loss_det + λ_adv * loss_adv + λ_id * loss_id + λ_cycle * loss_cycle

        # 9. Backprop 
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # 10. Step schedulers
        scheduler_G.step(epoch + batch_index / total_batches)
        scheduler_D.step(epoch + batch_index / total_batches)

        # 11. Accumulate losses for the epoch
        epoch_loss_G += loss_G.detach().item()
        epoch_loss_D += loss_D.detach().item()
        epoch_loss_det += loss_det.detach().item()
        epoch_loss_adv += loss_adv.detach().item()
        epoch_loss_id += loss_id.detach().item()
        epoch_loss_cycle = loss_cycle.detach().item()

    # Log epoch losses
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss G_total: {epoch_loss_G:.4f}, "
          f"Loss G_adv: {epoch_loss_adv:.4f}, "
          f"Loss G_detect: {epoch_loss_det:.4f}, "
          f"Loss G_identity: {epoch_loss_id:.4f}, "
          F"Loss G_cycle: {epoch_loss_cycle:.4f}, "
          f"Loss D: {epoch_loss_D:.4f}, "
          f"{'WARMUP' if epoch < warmup_epochs else 'FULL'}")

    if (epoch + 1) % 5 == 0:
        torch.save(G_AB.state_dict(), f"/home/umang.shikarvar/CycleGAN/delhi2lucknow/generator_AB_epoch_{epoch+1}.pth")
        torch.save(G_BA.state_dict(), f"/home/umang.shikarvar/CycleGAN/lucknow2delhi/generator_BA_epoch_{epoch+1}.pth")