:::mermaid
flowchart TB
  %% ──────────── 1. OBSERVATION WINDOW ────────────
  subgraph OBS["Observation window (T = -5s to 0s, 50 steps)"]
    tgt_hist["Target agent 50 x 6 features"]
    oth_hist["49 other agents 50 x 6 features"]
    map_poly["HD-map polylines. 
    Install av2, add maps 30% gain MSE"]
  end

  %% ──────────── 2. ENCODERS ────────────
  tgt_hist --> enc_tgt["LSTM / GRU history encoder"]
  oth_hist --> enc_oth["LSTM / GRU (shared weights)"]
  map_poly --> map_enc["LaneGCN / GNN map encoder"]

  %% ──────────── 3. SCENE FUSION ────────────
  enc_tgt & enc_oth & map_enc --> fusion["Transformer cross-attention fusion. 
  compress 5s history per agent per scene into fixed length vector"]
  fusion --> ctx_vec["Context vector (128-256 D)"]

  %% ──────────── 4. TRAJECTORY DECODER ────────────
  subgraph DEC["Trajectory decoder (future 6 s)"]
    mode_q["Mode queries M = 6"]
    state_q["State queries T = 60"]
    ctx_vec --> mode_q
    ctx_vec --> state_q
    mode_q --> temp_dec["Temporal decoder"]
    state_q --> temp_dec
  end

  %% ──────────── 5. POST-PROCESSING ────────────
  temp_dec --> proposals["Trajectory proposals (M x 60 x 2)"]
  proposals --> refine["Anchor refinement + kinematic filter"]
  refine --> nms["NMS / clustering keep K = 6"]
  nms --> final["Final K trajectories (60 x 2 each)"]

  %% ──────────── 6. (OPTIONAL) ENSEMBLE ────────────
  final -.-> ensemble["Offline ensemble & averaging"]
  ensemble -.-> submit["Submission / evaluation"]

:::