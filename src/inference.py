import torch 
from pipline import TrajectoryForecaster

if __name__ == "__main__":
    # Configuration
    cfg = {
        "encoder": {"in_dim":16,"hidden_dim":128,"num_layers":1,"type_emb_dim":8,"num_types":10},
        "social": {"enabled":True,"input_dim":128,"heads":4},
        "decoder": {"type":"mlp","input_dim":128,"output_steps":60},
        "refiner": {"enabled":False,"hidden_dim":128},
        "batch_size":32,"lr":1e-3,"epochs":5
    }
    # Dummy data
    scenes = torch.randn(100,50,110,6)
    targets = torch.randn(100,60,2)
    dataset = EgoDataset(scenes, targets)
    # Build and train
    model = TrajectoryForecaster(cfg)
    trained_model = train_model(model, dataset, cfg)

    # Inference on a single batch
    hist, neigh, tgt = dataset[0]
    with torch.no_grad():
        pred = trained_model(hist.unsqueeze(0), neigh.unsqueeze(0))
    print("Inference output shape:", pred.shape)