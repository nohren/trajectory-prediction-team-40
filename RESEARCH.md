# Research

## 2D Data

For each of the 50 agents in a scene, there are 660 (110 * 6) 6 tupple blocks of data describing the agents 11 seconds of history+future.  5s history + 6 seconds future in the training set. The ego vehicle is always index 0.

Train - `(10000,50,110,6)`

Validation - `80/20 split, 8000 train, 2000 validate`

Test - `(2100, 50, 50, 6)`

Task: predict the path of agent[0], the ego agent vehicle, over the next 6 seconds given the previous 5.  No map data is given, we can use context from social interactions with other objects.

Output - predict the next 6 seconds of positions $(\hat{x},\hat{y})$ for the ego agent across all scenes and frames `(2100*60,2)`

Evaluation - Mean squared error of final trajectory using L2 distance where n = frames = 60. $\frac{1}{n}\sum^n_{i=1}[(x_i - \hat{x_i})^2 + (y_i - \hat{y_i})^2]$

### 6 tupple data block

### position
`position_x: float - x-coordinate of the agent's position`
`position_y: float - y-coordinate of the agent's position`

### velocity
`velocity_x: float - x-component of the agent's velocity`
`velocity_y: float - y-component of the agent's velocity`

### heading
`heading: float - heading angle of the agent in radians`

### object type

`object_type: int - encoded object type. int indexes this array:
['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background', 'construction', 'riderless_bicycle', 'unknown']`
- note it got cast to float32 on export, need to cast it back to int on import 

## Inductive Biases
1. [✅] Translation / Rotation invariance bias - **data preprocessing step** - the bias is that the action of rotation or translation is what care we about, not the headings or absolute locations. This reduces variability the model must handle.  Instead of seeing the same maneuver in hundreds of global orientations/locations, it sees it in one canonical frame. 

2. [✅] Temporal (time based) smooth changes bias - sampling at 10Hz or 1 sample every 1/10th of a second.  We are using the previous 50 samples to predict the next 60.  Transformer/GRU/LSTM/1D-CNN are designed to model sequential data efficiently. This can encode a vector representation (50x6) of each agent to predict the next 60.

3. [⛔] Road network bias - Vehicle motion constrained by the road. **We don't have map data** so nothing to do here. If we did, we could use vecotrized map encoders (laneGCN, TNT). Road context can improve MSE by 30-40%.

4. [✅] Social bias - nearby agents influence each other more than distance ones.  Use self atttention to learn which neighboring nodes are more important when aggregating information. Graph attention Network (GAT) / Graph Neural Network (GNN) /  / HiVT

5. [ ] Multi-Modal bias - given past there are many futures possible.  Anchor based - multipath++, query based - QCNet.

6. [ ] Kinematic & Physical constraints Bias - vehicles must abide by the laws of physics. Vehicle acceleration, turn rates, and curvature of turns is bounded.  SIMPL uses polynomial parameterization with bernstein polynomials for efficiency. Post-hoc filtering clip predictions to physics boundaries.

7. [⛔] Scene centric consistency bias - All agents futures jointly plausible.  This bias is mainly used for multi-agent forcasting.  **We are single-agent forcasting**. FJMP - factorial DAG decoder predicts agents in topological order each conditioning on the previous. Joint transformers used in multi-agent forcasting.

## An example of something created by chat gpt research based on current SOTAS
Chat GPT's frankenstein 
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