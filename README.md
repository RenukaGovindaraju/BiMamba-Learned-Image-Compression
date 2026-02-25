# 🧠 Image Compression and Decompression

This project implements a Mamba-based deep learning framework for efficient image compression and decompression. The model is trained to minimize distortion while maximizing compression efficiency.

---

## 📊 Validation Results

The following validation graphs are generated to evaluate the model performance:

- **BPP vs PSNR**
- **BPP vs Loss**

You can find the plot here:  
📁 `validation_bpp_curves.png`

---

## 🖼️ Reconstructed Images

The reconstructed images demonstrate how well the model preserves visual quality after compression. Some sample images are:

- `kodim01.png`
- `kodim04.png`
- `kodim07.png`

All reconstructed output images are located in the folder:  
�� `reconstructed_images/`

These images were generated using the **best trained model checkpoint**:

📍 `experiments/Mamba_Final_Correct_Path/checkpoints/best_model_e081_loss0.4181.pth.tar`

> ℹ️ **Note**: This model file exceeds GitHub's upload limit of 100MB, so it is hosted externally.

---

## �� Download Best Model

You can download the best model checkpoint from this link:  
👉 [Download best_model_e081_loss0.4181.pth.tar](https://drive.google.com/your-shareable-link)

---

## 📦 Folder Structure


---

## 🧪 Evaluation Metrics

The model performance is evaluated using:

- **PSNR (Peak Signal-to-Noise Ratio)**
- **MS-SSIM (Multi-Scale Structural Similarity Index)**
- **Compression Ratio (BPP)**

---

## ✍️ Citation

If you use this project in your research or publication, please cite it appropriately.
