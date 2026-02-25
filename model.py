import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import MaskedConv2d

# Try to import Mamba, exit if missing
try:
    from mamba_ssm import Mamba
except ImportError:
    print("FATAL ERROR: mamba_ssm is not installed. Please run 'pip install mamba-ssm'")
    exit()

class BiMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        # Bidirectional Mamba: One forward, one backward
        self.mamba_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.linear_project = nn.Linear(d_model * 2, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        identity = x
        
        # 1. Flatten for Sequence Modeling: [B, C, H*W] -> [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        
        # 2. Forward Direction
        out_fwd = self.mamba_fwd(x_norm)
        
        # 3. Backward Direction (Flip, Process, Flip Back)
        x_rev = torch.flip(x_norm, dims=[1])
        out_bwd = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # 4. Concatenate and Project
        combined = torch.cat([out_fwd, out_bwd], dim=-1) # [B, L, 2*C]
        out_merged = self.linear_project(combined)       # [B, L, C]
        out_final = self.norm2(out_merged)
        
        # 5. Reshape back to Image: [B, L, C] -> [B, C, H, W]
        out_reshaped = out_final.transpose(1, 2).view(B, C, H, W)
        
        return identity + out_reshaped

class MyHierarchicalEncoder(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2), BiMambaBlock(N), 
            conv(N, N, kernel_size=5, stride=2), BiMambaBlock(N), 
            conv(N, N, kernel_size=5, stride=2), BiMambaBlock(N),
            conv(N, M, kernel_size=5, stride=2),
        )
    def forward(self, x): return self.g_a(x)

class MyHierarchicalDecoder(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2), BiMambaBlock(N),
            deconv(N, N, kernel_size=5, stride=2), BiMambaBlock(N),
            deconv(N, N, kernel_size=5, stride=2), BiMambaBlock(N),
            deconv(N, 3, kernel_size=5, stride=2),
        )
    def forward(self, x): return self.g_s(x)

class Hyperprior(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3), nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
    def forward(self, y):
        z = self.h_a(torch.abs(y))
        return z, self.h_s(z)

class ContextModel(nn.Module):
    def __init__(self, M, N):
        super().__init__()
        self.context_prediction = nn.Sequential(
            MaskedConv2d(M, N, kernel_size=5, padding=2, stride=1), nn.LeakyReLU(inplace=True),
            MaskedConv2d(N, N, kernel_size=5, padding=2, stride=1), nn.LeakyReLU(inplace=True),
            MaskedConv2d(N, M * 2, kernel_size=5, padding=2, stride=1),
        )
    def forward(self, y_hat): return self.context_prediction(y_hat)

class MySOTAMambaModel(CompressionModel):
    def __init__(self, N=192, M=320):
        super().__init__() 
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.g_a = MyHierarchicalEncoder(N, M)
        self.g_s = MyHierarchicalDecoder(N, M)
        self.hyperprior = Hyperprior(N, M)
        self.context_model = ContextModel(M, N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        z, hyper_params = self.hyperprior(y)
        
        # Entropy Bottleneck (z)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        # Hyperparameters (y)
        hyper_scales_hat, hyper_means_hat = hyper_params.chunk(2, 1)
        
        # Context Model
        y_hat_quant = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        context_params = self.context_model(y_hat_quant)
        context_scales, context_means = context_params.chunk(2, 1)
        
        # Gaussian Parameters
        final_scales = torch.clamp(hyper_scales_hat + context_scales, min=1e-3, max=10.0)
        final_means = hyper_means_hat + context_means
        
        # Probability estimation (y)
        y_hat, y_likelihoods = self.gaussian_conditional(y, final_scales, means=final_means)
        
        # Decoder
        x_hat = self.g_s(y_hat)
        
        return {
            "x_hat": x_hat, 
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }
