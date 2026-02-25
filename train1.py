import argparse
import math
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

from model import MySOTAMambaModel

class ResizeIfTooSmall:
    def __init__(self, size=256):
        self.size = size
    def __call__(self, img):
        w, h = img.size
        if w < self.size or h < self.size:
            return TF.resize(img, self.size)
        return img

class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [
            os.path.join(root, f) for f in os.listdir(root) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in {root}. Check path.")
        print(f"Loaded {len(self.images)} images from {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def configure_optimizers(net, args):
    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    params_dict = dict(net.named_parameters())
    
    optimizer = optim.Adam((params_dict[n] for n in sorted(list(parameters))), lr=args.learning_rate)
    # Kept stable Aux LR
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(list(aux_parameters))), lr=3e-4)
    return optimizer, aux_optimizer

# --- FIX 3: LAMBDA WARMUP ---
class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.base_lmbda = lmbda

    def forward(self, output, target, step=0):
        # Warmup logic: 0 to 100% over 5000 steps
        lmbda = self.base_lmbda
        if step < 5000:
             lmbda = self.base_lmbda * (step / 5000.0)
             # Prevent lambda being exactly 0
             lmbda = max(lmbda, 1e-6)

        N, _, H, W = target.size()
        num_pixels = N * H * W
        
        probs_y = torch.clamp(output["likelihoods"]["y"], min=1e-9)
        probs_z = torch.clamp(output["likelihoods"]["z"], min=1e-9)
        
        bpp_loss = (torch.log(probs_y).sum() + torch.log(probs_z).sum()) / (-math.log(2) * num_pixels)
        mse_loss = self.mse(output["x_hat"], target)
        
        rd_loss = lmbda * (255**2) * mse_loss + bpp_loss
        return rd_loss, bpp_loss, mse_loss

def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_psnr = 0
    steps = len(val_loader)
    with torch.no_grad():
        for d in val_loader:
            d = d.to(device)
            h, w = d.size(2), d.size(3)
            pad_h = (64 - h % 64) % 64; pad_w = (64 - w % 64) % 64
            d_pad = nn.functional.pad(d, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else d
            
            out = model(d_pad)
            x_hat = out["x_hat"][:, :, :h, :w]
            mse = nn.functional.mse_loss(x_hat, d)
            psnr = 10 * math.log10(1. / mse.item())
            
            probs_y = torch.clamp(out["likelihoods"]["y"], min=1e-9)
            probs_z = torch.clamp(out["likelihoods"]["z"], min=1e-9)
            bpp = (torch.log(probs_y).sum() + torch.log(probs_z).sum()) / (-math.log(2) * h * w)
            
            # Validation always uses full lambda (no warmup) to check real progress
            val_loss += (criterion.base_lmbda * 255**2 * mse + bpp).item()
            val_psnr += psnr
    return val_loss / steps, val_psnr / steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-v", "--val-dataset", required=True)
    parser.add_argument("--lambda", type=float, required=True, dest="lmbda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-path", default="check_points")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} | Lambda: {args.lmbda} | Batch Size: {args.batch_size}")
    
    train_tf = transforms.Compose([
        ResizeIfTooSmall(256),
        transforms.RandomCrop(256), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()
    ])
    val_tf = transforms.Compose([transforms.ToTensor()])

    train_set = FlatImageFolder(args.dataset, transform=train_tf)
    val_set = FlatImageFolder(args.val_dataset, transform=val_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    net = MySOTAMambaModel(N=192, M=320).to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)
    scaler = GradScaler()

    save_dir = os.path.join(args.save_path, f"lambda_{args.lmbda}")
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    
    # Global step tracker for warmup
    global_step = 0

    for epoch in range(args.epochs):
        net.train()
        for i, d in enumerate(train_loader):
            d = d.to(device)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            
            with autocast():
                # Pass global_step to criterion
                out = net(d)
                loss, bpp, mse = criterion(out, d, step=global_step)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            aux_loss = net.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            
            global_step += 1
            
            if i % 1000 == 0:
                print(f"Ep {epoch} | It {i} | Loss: {loss.item():.4f} | BPP: {bpp.item():.4f} | MSE: {mse.item():.5f}")

        v_loss, v_psnr = validate(net, criterion, val_loader, device)
        scheduler.step(v_loss)
        print(f"==> Val Ep {epoch} | Loss: {v_loss:.4f} | PSNR: {v_psnr:.2f} dB")
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(net.state_dict(), os.path.join(save_dir, "best.pth"))
            print(f"   [SAVED] Best model: {v_loss:.4f}")

if __name__ == "__main__":
    main()
    
