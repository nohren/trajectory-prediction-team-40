import os, re, glob, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------
# UTILITIES: Transforms
# -----------------------


# translation and rotation invariance for ego (x,y) training loop y ground truth comparison
def align_future(future, center, theta):
    xy = future - center
    c, s = np.cos(-theta), np.sin(-theta)
    x_new = xy[..., 0] * c - xy[..., 1] * s
    y_new = xy[..., 0] * s + xy[..., 1] * c
    return np.stack([x_new, y_new], axis=-1)


# converts to a relative coordinate system, so model can focus on the patterns
def invariance_transform(past, accel_dt=None):
    """
    Convert a scene to an ego-centric, velocity-aligned frame.

    Args
    ----
    past : (A, T, 6) float array
        [:,:,0:2] = x, y
        [:,:,2:4] = vx, vy
        [:,:,4]   = heading  (rad)
        [:,:,5]   = type_id  (int)
    accel_dt : float or None
        Sampling period in seconds. If given, two acceleration channels (ax, ay)
        are added so output has 9 features. If None, acceleration is omitted and
        output has 7 features.

    Returns
    -------
    aligned : (A, T, 7 or 9) float array
    center  : (2,)            ego’s last (x,y) in world frame
    theta   : float           ego’s last heading (rad)
    """
    A, T, F = past.shape
    assert F == 6, f"expected feat_dim = 6, got {F}"

    pos = past[..., 0:2]  # (A,T,2)
    vel = past[..., 2:4]  # (A,T,2)
    heading = past[..., 4]  # (A,T)
    obj_id = past[..., 5].astype(int)  # keep as float for stacking

    # --- translate so ego’s last position is origin ------------------
    center = pos[0, -1].copy()  # (2,)
    pos_t = pos - center  # broadcasting (A,T,2) - (2,)

    # --- rotate so ego’s last heading is +X --------------------------
    theta = heading[0, -1]  # scalar
    c, s = np.cos(-theta), np.sin(-theta)  # rotation matrix

    # Rotate pos and vel
    R = np.array([[c, -s], [s, c]])  # 2×2
    pos_r = pos_t @ R.T  # (A,T,2)
    vel_r = vel @ R.T  # (A,T,2)

    # --- optional acceleration --------------------------------------
    if accel_dt is not None:
        inv_dt = 1.0 / accel_dt
        acc_r = np.zeros_like(vel_r)
        acc_r[:, 1:] = (vel_r[:, 1:] - vel_r[:, :-1]) * inv_dt
        features = 9
    else:
        features = 7

    # --- relative heading -------------------------------------------
    heading_rel = ((heading - theta + np.pi) % (2 * np.pi)) - np.pi  # (A,T)
    heading_cos = np.cos(heading_rel)  # (A,T)
    heading_sin = np.sin(heading_rel)  # (A,T)
    heading_vec = np.stack([heading_cos, heading_sin], axis=-1)  # (A,T,2)

    # --- stack output -----------------------------------------------
    aligned = np.zeros((A, T, features), dtype=past.dtype)
    aligned[..., 0:2] = pos_r
    aligned[..., 2:4] = vel_r
    if accel_dt is not None:
        aligned[..., 4:6] = acc_r
        aligned[..., 6:8] = heading_vec
        aligned[..., 8] = obj_id
    else:
        aligned[..., 4:6] = heading_vec
        aligned[..., 6] = obj_id

    return aligned, center, theta


# batch inverse transform for use outside in prediction, we only care about position inverse rotation and translation
def inverse_transform(pred, centers, thetas):
    """
    Bring aligned predictions back to world coordinates.

    Args
    ----
    pred    : (..., T, 2)  aligned positions
    centers : (..., 2)     translation(s) subtracted in forward pass
    thetas  : (...)        rotation angle(s) (rad), same leading dims as centers

    Returns
    -------
    world : (..., T, 2)  positions in world frame
    """
    pred = np.asarray(pred)
    centers = np.asarray(centers)
    thetas = np.asarray(thetas)

    # Bring everything to shape (..., T, 2)
    # Allow leading batch dims of arbitrary rank
    # Broadcasting handles scalars automatically.
    c = np.cos(thetas)[..., None]  # (..., 1)
    s = np.sin(thetas)[..., None]  # (..., 1)

    # Rotate back
    x, y = pred[..., 0], pred[..., 1]
    x_w = x * c - y * s
    y_w = x * s + y * c
    world = np.stack([x_w, y_w], axis=-1)  # (..., T, 2)

    # Translate back
    world += centers[..., None, :]  # broadcast center over T

    return world


# -----------------------
# COLLATE FUNCTION
# -----------------------


def collate_fn(batch):
    l = len(batch[0])
    if l == 5:
        # train: (past,mask,future)
        pasts, masks, futures, centers, thetas = zip(*batch)
        return (
            torch.stack(pasts),
            torch.stack(masks),
            torch.stack(futures),
            torch.stack(centers),  # shape (B,2)
            torch.stack(thetas),  # shape (B,)
        )
    elif l == 4:
        # test: (past,mask,center,theta)
        pasts, masks, centers, thetas = zip(*batch)
        return (
            torch.stack(pasts),
            torch.stack(masks),
            torch.stack(centers),  # shape (B,2)
            torch.stack(thetas),  # shape (B,)
        )
    else:
        raise ValueError(f"Unrecognized sample of length {l}")


# -----------------------
# DATASET
# -----------------------


class TrajectoryDataset(Dataset):
    def __init__(self, cfg, input_path=None, data=None, is_test=False):
        if data is not None:
            self.data = data
        else:
            npz = np.load(input_path)
            self.data = npz["data"]

        # scene0 = self.data[0]  # (A, T, F)
        # print("feature dim  :", scene0.shape[-1])
        # print("unique vals in col 5:", np.unique(scene0[..., 5]))
        # print("unique vals in last :", np.unique(scene0[..., -1]))

        self.T_past = cfg["data"]["T_past"]
        self.T_future = cfg["data"]["T_future"]
        self.accel_dt = cfg["data"]["accel_dt"]
        self.is_test = is_test

        # Calculate normalization statistics from the past data
        self.calculate_normalization_stats()

    def calculate_normalization_stats(self):
        """Calculate mean and std for efficient normalization"""
        # align past data
        all_pos, all_vel, all_acc = [], [], []
        for scene in self.data:
            past = scene[:, : self.T_past, :].copy()
            past_aligned, _, _ = invariance_transform(past, self.accel_dt)

            # collect positions & velocities across all agents & all time-steps
            all_pos.append(past_aligned[..., :2].reshape(-1, 2))
            all_vel.append(past_aligned[..., 2:4].reshape(-1, 2))

            # collect acceleration if present
            if self.accel_dt is not None and past_aligned.shape[-1] >= 6:
                all_acc.append(past_aligned[..., 4:6].reshape(-1, 2))

        # mask is there to exclude padded zeros used in trajectory data
        # from the mean-/std-computation and
        # to handle the corner-case where the slice being analysed contains nothing but padding

        # ---------- positions ----------
        all_pos = np.concatenate(all_pos, axis=0)
        mask = np.abs(all_pos).sum(-1) > 0
        if mask.any():
            valid_pos = all_pos[mask]
            self.pos_mean = valid_pos.mean(0)
            self.pos_std = np.maximum(valid_pos.std(0), 1e-6)
        else:
            self.pos_mean = np.zeros(2)
            self.pos_std = np.ones(2)

        # ---------- velocities ----------
        all_vel = np.concatenate(all_vel, axis=0)
        mask = np.abs(all_vel).sum(-1) > 0
        if mask.any():
            valid_vel = all_vel[mask]
            self.vel_mean = valid_vel.mean(0)
            self.vel_std = np.maximum(valid_vel.std(0), 1e-6)
        else:
            self.vel_mean = np.zeros(2)
            self.vel_std = np.ones(2)

        # ---------- accelerations ----------
        if all_acc:  # list not empty
            all_acc = np.concatenate(all_acc, axis=0)
            mask = np.abs(all_acc).sum(-1) > 0
            if mask.any():
                valid_acc = all_acc[mask]
                self.acc_mean = valid_acc.mean(0)
                self.acc_std = np.maximum(valid_acc.std(0), 1e-6)
            else:
                self.acc_mean = np.zeros(2)
                self.acc_std = np.ones(2)
        else:
            # if accel_dt is None we still define attrs so code downstream is safe
            self.acc_mean = np.zeros(2)
            self.acc_std = np.ones(2)

    # (x − μ) / σ
    def normalize_features(self, features):
        """Normalize features efficiently"""
        normalized = features.copy()
        # Normalize positions (x, y)
        normalized[..., 0:2] = (features[..., 0:2] - self.pos_mean) / self.pos_std
        # Normalize velocities (vx, vy)
        normalized[..., 2:4] = (features[..., 2:4] - self.vel_mean) / self.vel_std
        # Normalize acceleration (ax, ay) #
        if self.accel_dt is not None:
            normalized[..., 4:6] = (features[..., 4:6] - self.acc_mean) / self.acc_std

        return normalized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]  # (num_agents, T, features) per scene calculations

        # Extract past trajectory
        past = scene[:, : self.T_past, :].copy()  # (num_agents, T_past, features)

        # Apply translation + rotation invariance per scene
        # (shifts ego → origin & rotates so ego’s heading is +x)
        past_aligned, center, theta = invariance_transform(past, self.accel_dt)

        # Normalize features
        past_aligned_normalized = self.normalize_features(past_aligned)

        # Create mask for valid agents (based on position)
        mask = np.sum(np.abs(past[:, :, :2]), axis=(1, 2)) > 0

        # training loss is based on predicting the aligned + normalized future ego vehicle trajectory
        if not self.is_test and scene.shape[1] >= self.T_past + self.T_future:
            future_raw = scene[
                0, self.T_past : self.T_past + self.T_future, :2
            ]  # Ego vehicle future (x, y)
            # align future ego to the same reference frame, then normalize
            future_aligned = align_future(future_raw, center, theta)
            future_aligned_normalized = (future_aligned - self.pos_mean) / self.pos_std

            return (
                torch.tensor(past_aligned_normalized, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(future_aligned_normalized, dtype=torch.float32),
                torch.tensor(
                    center, dtype=torch.float32
                ),  # needed to de-align per scene
                torch.tensor(theta, dtype=torch.float32),
            )

        # For test data, only return aligned and normalized past
        return (
            torch.tensor(past_aligned_normalized, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(center, dtype=torch.float32),  # needed to de-align per scene
            torch.tensor(theta, dtype=torch.float32),  # needed to de-align per scene
        )

    def denormalize_prediction(self, prediction):
        """Convert normalized predictions back to original scale"""
        return prediction * self.pos_std + self.pos_mean


# -----------------------
# MODEL: Transformer
# -----------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class AgentTypeEmbedding(nn.Module):
    def __init__(self, num_types=10, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(num_types, d_model)

    def forward(self, x):
        obj_type = x[..., -1].long()
        return self.embedding(obj_type)


class ImprovedTrajectoryTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg["model"]  # shorthand
        D = cfg["data"]  # shorthand

        feature_dim = M["feature_dim"]  # int, not tuple
        d_model = M["d_model"]
        nhead = M["nhead"]
        num_layers = M["num_layers"]
        dim_feedforward = M["dim_feedforward"]
        dropout = M["dropout"]
        T_past = D["T_past"]
        T_future = D["T_future"]

        self.d_model = d_model
        self.T_past = T_past
        self.T_future = T_future

        # Feature embedding for positions, velocities, accelerations
        self.feature_embed = nn.Linear(feature_dim, d_model)

        # Object type embedding
        self.type_embedding = AgentTypeEmbedding(num_types=10, d_model=d_model)

        # Positional encoding for timesteps
        self.pos_encoding = PositionalEncoding(d_model)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Transformer encoder for temporal relations
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_encoder_layer, num_layers=num_layers // 2
        )

        # Transformer encoder for social relations
        social_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.social_encoder = nn.TransformerEncoder(
            social_encoder_layer, num_layers=num_layers // 2
        )

        # Output MLP
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 2 * T_future),
        )

    def forward(self, past, agent_mask):
        B, N, T, F = past.shape  # Batch, Num_agents, Time, Features

        assert F >= 7, f"Expected at least 7 features, got {F}"

        # Embed all features directly
        features_flat = past.reshape(B * N * T, F)
        feature_embedding = self.feature_embed(features_flat)  # project to higher space
        feature_embedding = feature_embedding.reshape(B, N, T, self.d_model)

        # Get object type embedding
        type_embedding = self.type_embedding(past)  # B, N, T, d_model

        # Combine embeddings
        combined_embedding = feature_embedding + type_embedding

        # Reshape for temporal transformer: (T, B*N, d_model)
        temporal_input = combined_embedding.permute(2, 0, 1, 3).reshape(
            T, B * N, self.d_model
        )

        # Add positional encoding
        temporal_input = self.pos_encoding(temporal_input)

        # Apply temporal transformer
        temporal_output = self.temporal_encoder(temporal_input)

        # Get the last temporal state for each agent
        agent_features = temporal_output[-1].reshape(
            B, N, self.d_model
        )  # B, N, d_model

        # Make sure there's at least one valid agent per batch
        if (~agent_mask).all(dim=1).any():
            fallback_mask = agent_mask.clone()
            fallback_mask[:, 0] = True  # At least use ego vehicle
            agent_mask = torch.where(
                agent_mask.sum(dim=1, keepdim=True) == 0, fallback_mask, agent_mask
            )

        # Prepare for social transformer: (N, B, d_model)
        social_input = agent_features.permute(
            1, 0, 2
        )  # want agent features back in the first dim, not time

        # Apply social transformer with masking
        social_output = self.social_encoder(
            social_input, src_key_padding_mask=~agent_mask
        )

        # Extract ego vehicle embedding
        ego_embedding = social_output[0]  # B, d_model

        # Apply prediction head
        trajectory_flat = self.prediction_head(ego_embedding)  # B, 2*T_future

        # Reshape to (Batch, Time, XY)
        predictions = trajectory_flat.reshape(B, self.T_future, 2)

        return predictions


# ------------------------------
# AUTO-DIFFERENTIATION
# -------------------------------
def train_epoch(model, dataloader, optimizer, device, clip_grad=0.3):
    model.train()
    total_loss = 0.0
    criterion = nn.SmoothL1Loss()

    for batch in dataloader:
        past, mask, future, _, _ = [x.to(device) for x in batch]

        optimizer.zero_grad()
        pred = model(past, mask)

        loss = criterion(pred, future)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()
        total_loss += loss.item() * past.size(0)

    return total_loss / len(dataloader.dataset)


# ------------------------------
# SCORING METRICS
# -------------------------------


def evaluate_kaggle(model, val_loader, val_dataset, device):
    """
    Computes Kaggle score √(Σ‖p̂ – p‖²) over the *entire* val set.
    Returns: (kaggle_score, val_mse_norm)
    """
    model.eval()

    mse_loss = nn.MSELoss(reduction="mean")
    mse_sum = 0.0
    n_batches = 0

    sq_err_global = 0.0  # accumulate Σ‖p̂ – p‖²  in metres²

    with torch.no_grad():
        for past, mask, future, centers, thetas in val_loader:
            past, mask = past.to(device), mask.to(device)
            future = future.numpy()  # still normalised/aligned
            centers = centers.numpy()
            thetas = thetas.numpy()

            # -------- model forward (normalised, aligned) --------
            pred_norm = model(past, mask).cpu().numpy()

            # -------- denormalise (still aligned) ---------------
            pred_aligned = val_dataset.denormalize_prediction(pred_norm)
            fut_aligned = val_dataset.denormalize_prediction(future)

            # -------- de-align (world coords, metres) -----------
            pred_world = inverse_transform(pred_aligned, centers, thetas)
            gt_world = inverse_transform(fut_aligned, centers, thetas)

            # -------- Kaggle global score -----------------------
            diff = pred_world - gt_world  # (B,60,2)
            sq_err = (diff**2).sum()
            sq_err_global += sq_err  # scalar

            # -------- cheap in-frame MSE for monitoring ---------
            # mse_sum += mse_loss(
            #     torch.from_numpy(pred_norm), torch.from_numpy(future)
            # ).item()
            n_batches += 1

    kaggle_score = np.sqrt(sq_err_global)  # √Σ‖.‖²
    val_mse_norm = (
        None  # mse_sum / n_batches  # implement on world scale at some point if useful?
    )

    return kaggle_score, val_mse_norm


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in val_loader:
            past, mask, future, _, _ = [x.to(device) for x in batch]
            pred = model(past, mask)
            loss = criterion(pred, future)
            total_loss += loss.item() * past.size(0)

    return total_loss / len(val_loader.dataset)


def predict(model, test_loader, test_dataset, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for past, mask, centers, thetas in test_loader:
            # Move inputs to device
            past, mask = past.to(device), mask.to(device)

            # predict in normalized aligned space
            pred_norm = model(past, mask).cpu().numpy()

            # undo normalization (still aligned)
            pred_aligned = test_dataset.denormalize_prediction(pred_norm)

            # undo relative alignment -> output world coords
            pred_world = inverse_transform(
                pred_aligned, centers.numpy(), thetas.numpy()
            )

            all_preds.append(pred_world)

    return np.concatenate(all_preds, axis=0)


def get_latest_checkpoint(folder):
    files = glob.glob(os.path.join(folder, "ckpt_epoch_*.pt"))
    if not files:
        return None
    return max(files, key=lambda f: int(re.findall(r"ckpt_epoch_(\d+)", f)[0]))
