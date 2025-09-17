

from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.att(x), dim=1)
        return torch.sum(x * w, dim=1)

class InterChannelAttention1D(nn.Module):
    def __init__(self, channels: int, proj: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels, bias=False) if proj else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)                                  # [B,T,C]
        B, T, C = z.shape
        A = torch.matmul(z.transpose(1, 2), z) / max(1,T) # [B,C,C]
        A = torch.softmax(A / (C ** 0.5), dim=-1)
        y = torch.matmul(z, A)                            # [B,T,C]
        return self.proj(y) + x

class LSK1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, ks: List[int] = [3,7,15,31], stride: int = 2):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.branches = nn.ModuleList([nn.Conv1d(c_out, c_out, k, padding=k//2, groups=c_out, bias=False) for k in ks])
        self.fuse = nn.Conv1d(c_out * len(ks), c_out, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        hidden = max(4, c_out // 8)
        self.gate = nn.Sequential(nn.Conv1d(c_out, hidden, 1), nn.ReLU(True), nn.Conv1d(hidden, len(ks), 1))
        self.down = nn.AvgPool1d(2, stride=stride, ceil_mode=True) if stride>1 else nn.Identity()
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        feats = [br(x) for br in self.branches]
        cat = torch.cat(feats, dim=1)
        fused = self.fuse(cat)
        alpha = torch.softmax(self.gate(self.pool(fused)).squeeze(-1), dim=-1)  # [B,S]
        out = 0
        for i, f in enumerate(feats):
            out = out + alpha[:, i].view(-1,1,1) * f
        out = self.act(self.bn(out))
        return self.down(out)

class Mamba2Block1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model*2, bias=False)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model)
        self.a = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B,T,D = x.shape
        gate, val = self.in_proj(x).chunk(2, dim=-1)
        g = torch.sigmoid(gate)
        v = self.dw_conv(val.transpose(1,2)).transpose(1,2)
        a = torch.tanh(self.a)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        y = torch.zeros_like(v)
        for t in range(T):
            h = a*h + g[:,t,:]*v[:,t,:]
            y[:,t,:] = h
        return self.norm(x + self.out_proj(y))

class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
    def forward(self, x): return self.net(x)

class SwitchMoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 4, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(d_model, hidden) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.drop = nn.Dropout(dropout)
        self._aux_loss = torch.tensor(0.0)
    @torch.no_grad()
    def _lb(self, gates_softmax, top1_idx, eps=1e-9):
        N,E = gates_softmax.shape
        importance = gates_softmax.sum(dim=0)/(N+eps)
        load = torch.zeros(E, device=gates_softmax.device, dtype=gates_softmax.dtype)
        load.scatter_add_(0, top1_idx, torch.ones_like(top1_idx, dtype=gates_softmax.dtype))
        load = load/(N+eps)
        return E * (importance * load).sum()
    def forward(self, x):  # [B,T,D]
        B,T,D = x.shape
        xt = x.reshape(B*T, D)
        logits = self.gate(xt)
        gates = torch.softmax(logits, dim=-1)
        top1 = torch.argmax(gates, dim=-1)
        self._aux_loss = self._lb(gates, top1)
        y = torch.zeros_like(xt)
        for e_id, expert in enumerate(self.experts):
            m = (top1 == e_id)
            if m.any(): y[m] = expert(xt[m])
        return self.drop(y).reshape(B,T,D)
    def aux_loss(self): return self._aux_loss

class GLRUCell(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2*d_model, bias=False)
        self.state_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B,T,D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        y = torch.zeros_like(x)
        for t in range(T):
            gx, vx = self.in_proj(x[:,t,:]).chunk(2, dim=-1)
            u = torch.sigmoid(gx)
            cand = self.state_proj(vx)
            h = u*h + (1-u)*cand
            y[:,t,:] = h
        return self.norm(y)

class LocalAttention1D(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, window: int = 16):
        super().__init__()
        self.h, self.w = heads, window
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B,T,D = x.shape; H = self.h; d = D//H; w = self.w; scale = d**-0.5
        q = self.q(x).view(B,T,H,d); k = self.k(x).view(B,T,H,d); v = self.v(x).view(B,T,H,d)
        out = torch.zeros_like(q)
        for t in range(T):
            s = max(0, t-w); e = min(T, t+w+1)
            qi = q[:,t:t+1]; ki = k[:,s:e]; vi = v[:,s:e]
            attn = (qi*scale) @ ki.transpose(-2,-1)
            attn = torch.softmax(attn, dim=-1)
            out[:,t:t+1] = attn @ vi
        out = out.view(B,T,D)
        return self.norm(x + self.proj(out))

class GriffinBlock1D(nn.Module):
    def __init__(self, d_model: int, window: int = 16, ffn_mult: int = 4):
        super().__init__()
        self.glru = GLRUCell(d_model)
        self.local_attn = LocalAttention1D(d_model, heads=4, window=window)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model*ffn_mult), nn.GELU(), nn.Linear(d_model*ffn_mult, d_model))
        self.out_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.glru(x)
        x = self.local_attn(x)
        x = x + self.ffn(x)
        return self.out_norm(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, a, b):  # a<-b
        B,T,D = a.shape; H = self.nhead; d = D//H; scale = d**-0.5
        q = self.q(a).view(B,T,H,d); k = self.k(b).view(B,T,H,d); v = self.v(b).view(B,T,H,d)
        attn = (q @ k.transpose(-2,-1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).view(B,T,D)
        return self.ln(a + self.dropout(self.proj(out)))

class LowRankBilinearFusion(nn.Module):
    def __init__(self, d_model: int, k: int = 64):
        super().__init__()
        self.pr = nn.Linear(d_model, k, bias=False)
        self.pv = nn.Linear(d_model, k, bias=False)
        self.out = nn.Linear(k, d_model, bias=False)
        self.ln  = nn.LayerNorm(d_model)
    def forward(self, r_vec, v_vec):
        r = torch.tanh(self.pr(r_vec)); v = torch.tanh(self.pv(v_vec))
        return self.ln(self.out(r*v))

class FuseMoE1D(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 4, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.moe = SwitchMoE(d_model, n_experts, hidden, dropout)
        self.ln  = nn.LayerNorm(d_model)
    def aux_loss(self): return self.moe.aux_loss()
    def forward(self, x):
        y = self.moe(x.unsqueeze(1)).squeeze(1)
        return self.ln(y + x)

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / max(1, half-1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb  # [B, dim]

class CosineNoiseSchedule:
    def __init__(self, timesteps: int = 1000, s: float = 0.008):
        self.T = timesteps; self.s = s
        t = torch.linspace(0, timesteps, timesteps+1)
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1+s) * math.pi/2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.alphas_cumprod = alphas_cumprod.clamp(min=1e-5, max=0.99999)

    def sample(self, t: torch.Tensor, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 alpha_bar_t, sigma_t"""
        if device is None: device = t.device
        ab = self.alphas_cumprod[t.long()].to(device)
        return ab, (1 - ab)

class DiffusionHead(nn.Module):
    def __init__(self, d_model: int = 128, t_dim: int = 64, hidden: int = 256, timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.LayerNorm(d_model + t_dim),
            nn.Linear(d_model + t_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_model)
        )
        self.schedule = CosineNoiseSchedule(timesteps=timesteps)

    def forward(self, m_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = m_vec.shape
        device = m_vec.device
        t = torch.randint(low=1, high=self.timesteps, size=(B,), device=device)
        alpha_bar, one_minus_alpha_bar = self.schedule.sample(t, device=device)  # [B], [B]
        alpha_bar = alpha_bar.view(B,1)
        sigma2 = one_minus_alpha_bar.view(B,1)

        eps = torch.randn_like(m_vec)               
        m_t = torch.sqrt(alpha_bar) * m_vec + torch.sqrt(sigma2) * eps 
        t_emb = timestep_embedding(t, self.t_dim).to(device)            # [B,t_dim]
        x_in = torch.cat([m_t, t_emb], dim=-1)                           # [B,D+t_dim]
        eps_pred = self.net(x_in)                                   
        return eps_pred, eps

class MultimodalP2MFDS_Final_Diffusion(nn.Module):
    def __init__(self,
                 radar_input_dim: int = 3,
                 vibration_input_dim: int = 3,
                 mid_channels: int = 128,
                 num_classes: int = 1,
                 moe_experts: int = 4,
                 moe_hidden: int = 256,
                 moe_dropout: float = 0.1,
                 vib_griffin_layers: int = 2,
                 vib_window: int = 16,
                 fusion_heads: int = 4,
                 mlb_k: int = 64,
                 diffusion_timesteps: int = 1000):
        super().__init__()
        # Radar
        self.radar_cnn = nn.Sequential(
            LSK1D(radar_input_dim, 64,  ks=[3,7,15,31], stride=2),
            LSK1D(64,               mid_channels, ks=[3,7,15,31], stride=2),
        )
        self.radar_seq  = Mamba2Block1D(mid_channels)
        self.radar_moe  = SwitchMoE(mid_channels, n_experts=moe_experts, hidden=moe_hidden, dropout=moe_dropout)

        # Vibration
        self.vibration_cnn = nn.Sequential(
            LSK1D(vibration_input_dim, 64,  ks=[3,7,15,31], stride=2),
            LSK1D(64,                 mid_channels, ks=[3,7,15,31], stride=2),
        )
        self.vibration_seq = nn.Sequential(*[GriffinBlock1D(mid_channels, window=vib_window) for _ in range(vib_griffin_layers)])
        self.vibration_ica = InterChannelAttention1D(mid_channels)

        # Cross-modal interaction + pooling
        self.ca_r_from_v = CrossAttentionBlock(mid_channels, nhead=fusion_heads)
        self.ca_v_from_r = CrossAttentionBlock(mid_channels, nhead=fusion_heads)
        self.pool_r = AttentionModule(mid_channels)
        self.pool_v = AttentionModule(mid_channels)

        # Fusion: MLB + FuseMoE
        self.mlb      = LowRankBilinearFusion(d_model=mid_channels, k=mlb_k)
        self.fuse_moe = FuseMoE1D(d_model=mid_channels, n_experts=moe_experts, hidden=moe_hidden, dropout=moe_dropout)

        # Classifier head
        self.fusion_head = nn.Sequential(nn.Linear(mid_channels, 128), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(128, num_classes))
        self.act = nn.Sigmoid() if num_classes == 1 else nn.Identity()

        # Diffusion head on m_vec
        self.diff_head = DiffusionHead(d_model=mid_channels, t_dim=64, hidden=256, timesteps=diffusion_timesteps)

    # aux getters
    def radar_moe_aux(self) -> torch.Tensor: return self.radar_moe.aux_loss()
    def fuse_moe_aux(self) -> torch.Tensor:  return self.fuse_moe.aux_loss()

    def forward(self,
                radar_input: torch.Tensor,
                vibration_input: torch.Tensor,
                use_diffusion: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Radar path
        r = radar_input.transpose(1, 2)
        r = self.radar_cnn(r)          # [B,128,T/4]
        r = r.transpose(1, 2)          # [B,T/4,128]
        r = self.radar_seq(r)
        r = self.radar_moe(r)

        # Vibration path
        v = vibration_input.transpose(1, 2)
        v = self.vibration_cnn(v)      # [B,128,T/4]
        v = v.transpose(1, 2)          # [B,T/4,128]
        v = self.vibration_seq(v)
        v = self.vibration_ica(v)

        # Cross-modal interaction
        r_enh = self.ca_r_from_v(r, v)
        v_enh = self.ca_v_from_r(v, r)

        # Pooling to vectors
        r_vec = self.pool_r(r_enh)     # [B,128]
        v_vec = self.pool_v(v_enh)     # [B,128]

        # Fusion
        m_vec = self.mlb(r_vec, v_vec) # [B,128]
        m_vec = self.fuse_moe(m_vec)   # [B,128]

        # Classifier
        logits = self.fusion_head(m_vec)   # [B,1] or [B,C]
        out = self.act(logits if self.act is not nn.Identity() else logits)

        # Optional: diffusion loss on m_vec (epsilon prediction)
        if use_diffusion:
            eps_pred, eps = self.diff_head(m_vec.detach()) 
            diff_loss = F.mse_loss(eps_pred, eps)
            return out, diff_loss
        else:
            return out, None
