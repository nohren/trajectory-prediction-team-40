import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pipeline import (
    ImprovedTrajectoryTransformer,
    TrajectoryDataset,
    train_epoch,
    get_latest_checkpoint,
    collate_fn,
    predict,
    evaluate,
    evaluate_kaggle,
)

device_count = torch.cuda.device_count()
print(device_count)  # Should print 2
for i in range(device_count):
    print(torch.cuda.get_device_name(i))  # Name of GPU i

# -----------------------
# CONFIGURATION
# -----------------------
cfg = {
    "paths": {
        "train_input": "data/train.npz",
        "test_input": "data/test_input.npz",
        "output_csv": "predictions.csv",
        "checkpoint_dir": "checkpoints",
        "checkpoint_path": "best_model.pt",
    },
    "data": {
        "T_past": 50,
        "T_future": 60,
        "accel_dt": 0.1,  # time between intervals for 10Hz signals is 0.1... set to None not to use acceleration
    },
    "model": {
        "feature_dim": 7,  # don't touch this, will be set automatically on line 55
        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.3,  # try 0.1
    },
    "training": {  # Hyperparameters
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 1000,
        "warm_up": 5,
        "patience": 70,
        "betas": (0.9, 0.999),
        "eta_min": 1e-6,
    },
}

# Automatically derive feature_dim from accel_dt: 7 without acceleration, 9 with
cfg["model"]["feature_dim"] = 7 + (2 if cfg["data"]["accel_dt"] is not None else 0)


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    # device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load some configurations
    output_csv = cfg["paths"]["output_csv"]
    checkpoints_dir = cfg["paths"]["checkpoint_dir"]
    checkpoint_path = cfg["paths"]["checkpoint_path"]
    os.makedirs(checkpoints_dir, exist_ok=True)

    # for tensorboard
    writer = SummaryWriter()

    # ------------------------------
    # DATA LOADING AND PREPROCESSING
    # -------------------------------

    # Train validation split
    full_data = np.load(cfg["paths"]["train_input"])["data"]
    # Split into train and validation (7:3)
    num_samples = len(full_data)
    num_train = int(0.8 * num_samples)
    # permute
    perm = np.random.permutation(num_samples)
    train_idx = perm[:num_train]
    val_idx = perm[num_train:]
    train_data = full_data[train_idx]
    val_data = full_data[val_idx]

    # Create datasets with normalization
    train_ds = TrajectoryDataset(cfg, data=train_data)
    val_ds = TrajectoryDataset(cfg, data=val_data)

    # Create test dataset using the same normalization stats as training
    test_ds = TrajectoryDataset(
        cfg, input_path=cfg["paths"]["test_input"], is_test=True
    )

    # Copy normalization stats from train_ds
    for ds in (val_ds, test_ds):
        ds.pos_mean = train_ds.pos_mean
        ds.pos_std = train_ds.pos_std
        ds.vel_mean = train_ds.vel_mean
        ds.vel_std = train_ds.vel_std
        ds.acc_mean = train_ds.acc_mean
        ds.acc_std = train_ds.acc_std

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ----------------------------------
    # CREATE MODEL, OPTIMIZER, SCHEDULER
    # ----------------------------------

    # Create model
    model = ImprovedTrajectoryTransformer(cfg)
    if device_count > 1:
        model = torch.nn.DataParallel(model)  # for multi-GPU
    model.to(device)

    # create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=cfg["training"]["betas"],
    )

    # create scheduler - cosine annealing per epoch with warmup
    warm_up_epochs = cfg["training"]["warm_up"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"] - warm_up_epochs,
        eta_min=cfg["training"]["eta_min"],
    )
    warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs else 1
        ),
    )

    # ----------------------------------
    # CONDUCT TRAINING
    # ----------------------------------

    # Training variables
    epochs = cfg["training"]["epochs"]
    start_epoch = 1
    best_val_loss = float("inf")
    no_improve_epochs = 0

    # Try to load checkpoint
    latest_ckpt = get_latest_checkpoint(checkpoints_dir)
    if latest_ckpt:
        print(f"Loading checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(
            f"✅ Resumed from epoch {start_epoch - 1} with val_loss={best_val_loss:.6f}"
        )

    # tensorboard writer start
    writer = SummaryWriter(log_dir="runs/exp1")  # creates runs/exp1/*

    # Training loop
    print(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, device)

        # evalute on Kaggle scoring metric
        kaggle_score, _ = evaluate_kaggle(model, val_loader, val_ds, device)

        # Update learning rate
        if epoch <= warm_up_epochs:
            warm_up_scheduler.step()
        else:
            scheduler.step()

        # Print progress
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}, Approx Kaggle Score: {kaggle_score:.3f}"
        )
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        # Save best model
        if val_loss <= best_val_loss and epoch > warm_up_epochs:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(
                f"✅ Best model saved at epoch {epoch} (val loss: {best_val_loss:.6f})"
            )
        else:
            no_improve_epochs += 1

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                f"{checkpoints_dir}/ckpt_epoch_{epoch:04d}.pt",
            )
            print(f"🧪 Checkpoint saved at {checkpoints_dir}/ckpt_epoch_{epoch:04d}.pt")

        # Early stopping
        if no_improve_epochs >= cfg["training"]["patience"]:
            print(f"Early stopping triggered after {epoch} epochs")
            break
        #     break
    writer.close()

    # Load best model for prediction
    print("Loading best model for prediction...")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    )

    # Generate predictions
    print("Generating predictions...")
    preds = predict(model, test_loader, test_ds, device)

    # Flatten predictions to match submission format (2100*60, 2)
    preds_flat = preds.reshape(-1, 2)

    # Create ID column (required for submission)
    ids = np.arange(len(preds_flat))

    # Save predictions to CSV
    output = np.column_stack((ids, preds_flat))
    header = "index,x,y"
    np.savetxt(
        output_csv,
        output,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.6f", "%.6f"],
    )
    print(f"Predictions saved to {output_csv}")
