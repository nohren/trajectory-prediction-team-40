import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# -----------------------
# Preprocess Function (Ego-Centric Stub)
# -----------------------
# def egocentric_transform(history, neighbors=None):
#     """
#     Pseudocode:
#     1) Extract ego's last position & heading from history[..., -1]
#     2) Build rotation matrix for -ego_heading
#     3) Subtract ego_pos from all positions
#     4) Rotate positions & velocities by the matrix
#     5) Retain heading and object_type
#     6) Return transformed tensors

# -----------------------
# Module Factories
# -----------------------

def build_encoder(cfg):
    return HistoryEncoder(
        input_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        type_emb_dim=cfg.type_emb_dim,
        num_types=cfg.num_types
    )

def build_social(cfg):
    if not cfg.get("enabled", False):
        return nn.Identity()
    if cfg.get("type", "attention") == "attention":
        return SocialAttention(input_dim=cfg.input_dim, heads=cfg.heads)
    return GraphAttention(input_dim=cfg.input_dim)

# Single-agent baseline (no anchor)
def build_decoder(cfg):
    if cfg.get("type", "mlp") == "mlp":
        return MLPDecoder(input_dim=cfg.input_dim, output_steps=cfg.output_steps)
    return QueryDecoder(d_model=cfg.d_model, num_modes=cfg.num_modes)


def build_refiner(cfg):
    if not cfg.get("enabled", False):
        return nn.Identity()
    return TrajectoryRefiner(hidden_dim=cfg.hidden_dim)

# -----------------------
# Module Implementations
# -----------------------

class HistoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, type_emb_dim, num_types):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, type_emb_dim)
        self.frame_mlp = nn.Sequential(
            nn.Linear(4 + 1 + type_emb_dim, input_dim),
            nn.ReLU()
        )  # pos(2) + vel(2) + heading(1) + type_emb
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):  # x: [batch, T, 6]
        pos = x[..., :2]
        vel = x[..., 2:4]
        head = x[..., 4:5]
        obj = x[..., 5].long()
        obj_emb = self.type_embed(obj)
        feat = torch.cat([pos, vel, head, obj_emb], dim=-1)
        ff = self.frame_mlp(feat)
        _, h = self.gru(ff)
        return h[-1]

class SocialAttention(nn.Module):
    def __init__(self, input_dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(input_dim, heads, batch_first=True)
    def forward(self, context, neighbors):
        seq = torch.cat([context.unsqueeze(1), neighbors], dim=1)
        out,_ = self.attn(seq, seq, seq)
        return out[:,0]

class GraphAttention(nn.Module):
    def __init__(self, input_dim): super().__init__(); self.fc=nn.Linear(input_dim, input_dim)
    def forward(self, context, neighbors): return self.fc(context)

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_steps):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(),
            nn.Linear(input_dim, output_steps * 2)
        )
        self.output_steps = output_steps
    def forward(self, context, anchors=None):
        b = context.size(0)
        out = self.net(context)
        return out.view(b, 1, self.output_steps, 2)

class QueryDecoder(nn.Module):
    def __init__(self, d_model, num_modes):
        super().__init__()
        self.mode_queries = nn.Parameter(torch.randn(num_modes, d_model))
        self.attn = nn.MultiheadAttention(d_model, heads=4, batch_first=True)
        self.out_fc = nn.Linear(d_model, 2)
    def forward(self, context, anchors=None):
        b,_ = context.size()
        keys = context.unsqueeze(1)
        queries = self.mode_queries.unsqueeze(0).expand(b,-1,-1)
        out,_ = self.attn(queries, keys, keys)
        preds = self.out_fc(out)
        return preds.unsqueeze(2).expand(-1,-1,60,-1)

class TrajectoryRefiner(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__(); self.net=nn.Sequential(
            nn.Conv1d(2,2,3,padding=1), nn.ReLU(), nn.Conv1d(2,2,3,padding=1)
        )
    def forward(self, preds, context=None):
        b,m,t,_ = preds.size()
        x = preds.view(b*m, t, 2).transpose(1,2)
        x = self.net(x)
        return x.transpose(1,2).view(b,m,t,2)

# -----------------------
# Main Model
# -----------------------

class TrajectoryForecaster(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder = build_encoder(cfg.encoder)
        self.social  = build_social(cfg.social)
        self.decoder = build_decoder(cfg.decoder)
        self.refiner = build_refiner(cfg.refiner)

    def forward(self, history, neighbors=None):
        enc = self.encoder(history)
        if neighbors is not None:
            b,k,t,f = neighbors.size(); flat=neighbors.view(b*k,t,f)
            n_enc = self.encoder(flat).view(b,k,-1)
            enc = self.social(enc, n_enc)
        preds = self.decoder(enc, None)
        return self.refiner(preds, enc)

# -----------------------
# Dataset and Training Loop
# -----------------------

class EgoDataset(Dataset):
    def __init__(self, scenes, targets):
        self.scenes = scenes  # [N,50,110,6]
        self.targets = targets  # [N,60,2]
    def __len__(self): return len(self.scenes)
    def __getitem__(self, idx):
        full = self.scenes[idx]  # (50,110,6)
        history = full[0,:50]    # [50,6]
        neighbors = full[1:,:50] # [49,50,6]
        target = self.targets[idx]
        return history, neighbors, target

def train_model(model, dataset, cfg):
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for hist, neigh, tgt in loader:
            preds = model(hist, neigh)
            loss = compute_min_mse_loss(preds, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * hist.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    return model
