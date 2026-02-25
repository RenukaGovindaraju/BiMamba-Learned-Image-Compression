import argparse
import os
import math
import time
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.utils import save_image
import glob
import io
from tqdm import tqdm

# Metrics
from pytorch_msssim import ms_ssim
import lpips

# Baselines
from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor

# Your Model
from model import MySOTAMambaModel

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    if mse == 0: return 100
    return -10 * math.log10(mse)

def pad(x, p=64):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    return F.pad(x, (0, W-w, 0, H-h), mode="reflect"), (0, W-w, 0, H-h)

def crop(x, padding):
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))

def run_jpeg(img_pil, quality):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    bytes_size = buffer.tell()
    buffer.seek(0)
    img_rec = Image.open(buffer).convert("RGB")
    return img_rec, bytes_size * 8 # Return image and bits

def make_visualization_strip(img_list, info_list, save_path):
    tensors = []
    try:
        # Standard Linux font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        font = ImageFont.load_default()

    for tensor, info_text in zip(img_list, info_list):
        # Move to CPU for Pillow
        pil_img = transforms.ToPILImage()(tensor.cpu().squeeze().clamp(0,1))
        
        # Create canvas with white footer for text
        footer_h = 170 
        canvas = Image.new("RGB", (pil_img.width, pil_img.height + footer_h), (255, 255, 255))
        canvas.paste(pil_img, (0,0))
        draw = ImageDraw.Draw(canvas)
        
        lines = info_text.split('\n')
        y = pil_img.height + 10
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            x = (canvas.width - text_w) // 2
            draw.text((x, y), line, font=font, fill="black")
            y += 30
            
        # Draw border
        draw.rectangle([(0,0), (canvas.width-1, canvas.height-1)], outline="black", width=2)
        tensors.append(transforms.ToTensor()(canvas))

    grid = torch.cat(tensors, dim=2)
    save_image(grid, save_path)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kodak", default="data/kodak", help="Path to Kodak")
    parser.add_argument("--tecnick", default="data/tecnick", help="Path to Tecnick")
    parser.add_argument("--ckpts", nargs='+', required=True, help="List of Mamba Checkpoints (Low to High)")
    parser.add_argument("--out_dir", default="phd_work2_safe", help="Output Folder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load LPIPS and keep on GPU
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    datasets = {"Kodak": args.kodak, "Tecnick": args.tecnick}
    all_results = []
    
    # VIS_CACHE: Stores baseline results to create the visual strip later
    # IMPORTANT: We store tensors on CPU to avoid CUDA OOM on Tecnick
    vis_cache = {}

    for ds_name, ds_path in datasets.items():
        if not os.path.exists(ds_path): continue
        print(f"\n>>> Processing {ds_name}...")
        images = sorted(glob.glob(os.path.join(ds_path, "*.*")))
        vis_cache[ds_name] = {}

        # ==================================================================
        # 1. RUN BASELINES (JPEG, Ballé, Cheng)
        # ==================================================================
        
        # --- A. JPEG ---
        print("   Running JPEG...")
        for img_path in tqdm(images, desc="JPEG"):
            img_name = os.path.basename(img_path)
            if img_name not in vis_cache[ds_name]: vis_cache[ds_name][img_name] = {}
            vis_cache[ds_name][img_name]["JPEG"] = []
            
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
            num_pixels = w*h
            x = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
            
            # Store Original info (CPU)
            orig_kb = (num_pixels * 3) / 1024
            vis_cache[ds_name][img_name]["Original"] = (x.cpu(), f"Original\nRaw Size: {orig_kb:.1f} KB\n24 bit/pixel")

            for q in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                rec_pil, bits = run_jpeg(pil_img, q)
                x_rec = transforms.ToTensor()(rec_pil).unsqueeze(0).to(device)
                
                bpp = bits / num_pixels
                psnr = compute_psnr(x, x_rec)
                msssim = ms_ssim(x, x_rec, data_range=1.0, size_average=True).item()
                
                # Compute LPIPS on GPU
                lp = lpips_fn((x-0.5)*2, (x_rec-0.5)*2).item()
                
                comp_kb = (bits / 8) / 1024
                cr = orig_kb / comp_kb if comp_kb > 0 else 0
                
                all_results.append([ds_name, "JPEG", q, img_name, bpp, psnr, msssim, lp, cr, orig_kb, comp_kb])
                
                # Cache for visuals (Store on CPU to save VRAM)
                info = f"JPEG (Q={q})\nBPP: {bpp:.3f} | PSNR: {psnr:.2f}\nSSIM: {msssim:.3f} | LPIPS: {lp:.3f}\nSize: {comp_kb:.1f} KB | CR: {cr:.1f}:1"
                vis_cache[ds_name][img_name]["JPEG"].append({'bpp': bpp, 'tensor': x_rec.cpu(), 'info': info})

        # --- B. DL Baselines ---
        for name, model_func in [("Balle2018", bmshj2018_hyperprior), ("Cheng2020", cheng2020_anchor)]:
            print(f"   Running {name}...")
            for q in [1, 2, 3, 4, 5, 6]:
                # Load model to GPU
                net = model_func(quality=q, pretrained=True).to(device).eval()
                net.update(force=True)
                
                for img_path in tqdm(images, desc=f"{name} Q={q}"):
                    img_name = os.path.basename(img_path)
                    if name not in vis_cache[ds_name][img_name]: vis_cache[ds_name][img_name][name] = []
                    
                    pil_img = Image.open(img_path).convert("RGB")
                    x = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
                    x_pad, padding = pad(x)
                    num_pixels = x.size(2) * x.size(3)
                    
                    with torch.no_grad():
                        out = net.compress(x_pad)
                        real_bits = sum(len(s)*8 for s_list in out["strings"] for s in s_list)
                        
                        out_dec = net.decompress(out["strings"], out["shape"])
                        x_rec = crop(out_dec["x_hat"], padding).clamp(0, 1)
                    
                    bpp = real_bits / num_pixels
                    psnr = compute_psnr(x, x_rec)
                    msssim = ms_ssim(x, x_rec, data_range=1.0, size_average=True).item()
                    lp = lpips_fn((x-0.5)*2, (x_rec-0.5)*2).item()
                    
                    orig_kb = (num_pixels * 3) / 1024
                    comp_kb = (real_bits / 8) / 1024
                    cr = orig_kb / comp_kb if comp_kb > 0 else 0
                    
                    all_results.append([ds_name, name, q, img_name, bpp, psnr, msssim, lp, cr, orig_kb, comp_kb])
                    
                    info = f"{name} (Q={q})\nBPP: {bpp:.3f} | PSNR: {psnr:.2f}\nSSIM: {msssim:.3f} | LPIPS: {lp:.3f}\nSize: {comp_kb:.1f} KB | CR: {cr:.1f}:1"
                    # Store on CPU
                    vis_cache[ds_name][img_name][name].append({'bpp': bpp, 'tensor': x_rec.cpu(), 'info': info})
                
                # Clear model to free VRAM for next iteration
                del net
                torch.cuda.empty_cache()

        # ==================================================================
        # 2. RUN PROPOSED MAMBA
        # ==================================================================
        print(f"   Running Proposed Mamba...")
        for i, ckpt in enumerate(args.ckpts):
            lam = os.path.basename(os.path.dirname(ckpt)) 
            net = MySOTAMambaModel(N=192, M=320).to(device)
            st = torch.load(ckpt, map_location=device)
            st = st['state_dict'] if 'state_dict' in st else st
            net.load_state_dict({k.replace("module.", ""): v for k, v in st.items()})
            net.eval()
            
            vis_dir_lam = os.path.join(args.out_dir, ds_name, "Visual_Comparisons", lam)
            os.makedirs(vis_dir_lam, exist_ok=True)

            for img_path in tqdm(images, desc=f"Mamba {lam}"):
                img_name = os.path.basename(img_path)
                pil_img = Image.open(img_path).convert("RGB")
                x = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
                x_pad, padding = pad(x)
                num_pixels = x.size(2)*x.size(3)
                
                with torch.no_grad():
                    out = net(x_pad)
                
                size_bits = sum(torch.log(l).sum() for l in out["likelihoods"].values()) / -math.log(2)
                real_bits = size_bits.item()
                
                x_rec = out["x_hat"].clamp(0, 1)
                x_final = crop(x_rec, padding).clamp(0, 1)
                
                bpp = real_bits / num_pixels
                psnr = compute_psnr(x, x_final)
                msssim = ms_ssim(x, x_final, data_range=1.0, size_average=True).item()
                lp = lpips_fn((x-0.5)*2, (x_final-0.5)*2).item()
                
                orig_kb = (num_pixels * 3) / 1024
                comp_kb = (real_bits / 8) / 1024
                cr = orig_kb / comp_kb if comp_kb > 0 else 0
                
                all_results.append([ds_name, "Proposed", lam, img_name, bpp, psnr, msssim, lp, cr, orig_kb, comp_kb])

                # ----------------------------------------------------------
                # GENERATE VISUAL STRIP (Matched BPP)
                # ----------------------------------------------------------
                def get_best_match(method_name):
                    candidates = vis_cache[ds_name][img_name].get(method_name, [])
                    if not candidates: return None
                    best = min(candidates, key=lambda z: abs(z['bpp'] - bpp))
                    return best

                match_jpeg = get_best_match("JPEG")
                match_balle = get_best_match("Balle2018")
                match_cheng = get_best_match("Cheng2020")
                
                if match_jpeg and match_balle and match_cheng:
                    info_proposed = f"Proposed\nBPP: {bpp:.3f} | PSNR: {psnr:.2f}\nSSIM: {msssim:.3f} | LPIPS: {lp:.3f}\nSize: {comp_kb:.1f} KB | CR: {cr:.1f}:1"
                    
                    img_list = [
                        vis_cache[ds_name][img_name]["Original"][0], # Already CPU
                        match_jpeg['tensor'], # Already CPU
                        match_balle['tensor'], # Already CPU
                        match_cheng['tensor'], # Already CPU
                        x_final.cpu() # Move to CPU
                    ]
                    info_list = [
                        vis_cache[ds_name][img_name]["Original"][1],
                        match_jpeg['info'],
                        match_balle['info'],
                        match_cheng['info'],
                        info_proposed
                    ]
                    
                    save_name = os.path.join(vis_dir_lam, f"Compare_{img_name}")
                    make_visualization_strip(img_list, info_list, save_name)
            
            # Clear Mamba model for next checkpoint
            del net
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Save CSVs
    # ----------------------------------------------------------------------
    cols = ["Dataset", "Method", "Quality", "Image", "BPP", "PSNR", "MSSSIM", "LPIPS", "CR", "OrigSize(KB)", "CompSize(KB)"]
    df = pd.DataFrame(all_results, columns=cols)
    df.to_csv(os.path.join(args.out_dir, "Detailed_Results.csv"), index=False)
    
    df_avg = df.groupby(["Dataset", "Method", "Quality"]).mean(numeric_only=True).reset_index()
    df_avg.to_csv(os.path.join(args.out_dir, "Average_Results.csv"), index=False)
    
    # ----------------------------------------------------------------------
    # Plot RD Curves
    # ----------------------------------------------------------------------
    print("\nGenerating Plots...")
    for ds in datasets.keys():
        subset = df_avg[df_avg["Dataset"] == ds]
        if subset.empty: continue
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for ax, metric in zip(axes, ["PSNR", "MSSSIM", "LPIPS"]):
            for method in ["JPEG", "Balle2018", "Cheng2020", "Proposed"]:
                data = subset[subset["Method"] == method].sort_values("BPP")
                if data.empty: continue
                ax.plot(data["BPP"], data[metric], marker='o', linewidth=2, label=method)
            
            ax.set_title(f"{ds} - {metric} vs Bitrate")
            ax.set_xlabel("Bitrate (bpp)")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{ds}_RD_Curves.png"))
        plt.close()

    print(f"\n>>> COMPLETE. Results saved in: {args.out_dir}")

if __name__ == "__main__":
    main()
