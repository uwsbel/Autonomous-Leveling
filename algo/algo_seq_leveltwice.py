import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import sys

# --- Project setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algo.nn_models import ForwardCNN, TinyForwardCNN, LightForwardCNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load model ===
def load_model(model_path):
    model = LightForwardCNN(action_dim=4).to(DEVICE)  # Changed to 4D action vector
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# === Predict full outputs ===
def predict_full_output(initial_heightmap: np.ndarray, control_action: np.ndarray, model_path: str):
    model = load_model(model_path)
    init_t = torch.tensor(initial_heightmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    act_t = torch.tensor(control_action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_hmap, pred_time = model(init_t, act_t)
    return pred_hmap.squeeze().cpu().numpy(), pred_time.item()

# === Optimization ===
def optimize_action(initial_heightmap: np.ndarray,
                   desired_heightmap: np.ndarray,
                   model_path: str,
                   lr: float = 1e-4,
                   num_iters: int = 500,
                   time_weight: float = 1e-7,
                   case: str = 'center',
                   scenario: str = 'init-mid'):
    """
    Optimize control actions for either init-mid or mid-final scenario
    Args:
        scenario: Either 'init-mid' or 'mid-final'
    """
    model = load_model(model_path)
    model.eval()

    # tensors
    desired_t = torch.tensor(desired_heightmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    init_t = torch.tensor(initial_heightmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Initialize action parameters (4D vector for v5 model)
    # init_action = np.array([0.1, -0.025, 0.1, -0.025], dtype=np.float32)
    init_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action_params = torch.tensor(init_action, dtype=torch.float32, requires_grad=True, device=DEVICE)
    optimizer = optim.Adam([action_params], lr=lr)

    mask = create_loss_mask(case, shape=(200, 100))
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    for it in range(1, num_iters + 1):
        optimizer.zero_grad()
        # Reshape action_params to have batch dimension
        action_params_reshaped = action_params.unsqueeze(0)  # Add batch dimension
        pred_hmap, pred_time = model(init_t, action_params_reshaped)

        # Heightmap loss
        diff = (pred_hmap - desired_t) * mask_t
        loss_h = torch.mean(diff ** 2)

        # Time term (encourage lower time)
        loss_time = pred_time.squeeze()
        total_loss = loss_h + time_weight * loss_time

        total_loss.backward()
        optimizer.step()

        # Clamp action values based on scenario
        if scenario == 'init-mid':
            # Clamp for init-mid (2 control pairs)
            for k in range(2):
                action_params.data[2 * k] = torch.clamp(action_params.data[2 * k], 0.0, np.pi / 6)
                action_params.data[2 * k + 1] = torch.clamp(action_params.data[2 * k + 1], -0.05, 0.075)
        else:  # mid-final
            # Clamp for mid-final (2 control pairs)
            for k in range(2):
                action_params.data[2 * k] = torch.clamp(action_params.data[2 * k], 0.0, np.pi / 6)
                action_params.data[2 * k + 1] = torch.clamp(action_params.data[2 * k + 1], -0.05, 0.075)

        if it % 100 == 0 or it == num_iters:
            vals = action_params.detach().cpu().numpy().reshape(2, 2)
            print(f"Iter {it:4d}/{num_iters} — Hmap Loss {loss_h.item():.6f} — Time {loss_time.item():.2f} — Total {total_loss.item():.6f}")
            print(f"Action: \n{vals}")

    return action_params.detach().cpu().numpy().squeeze()

# === Helpers ===
def create_loss_mask(case: str, shape):
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    def xy_to_idx(x, y):
        i = int((x / 4.0) * (H - 1))
        j = int(((y + 1.0) / 2.0) * (W - 1))
        return i, j

    if case == 'center':
        x1, y1, x2, y2 = 1.5, -0.5, 4.5, 0.5
    elif case == 'upper':
        x1, y1, x2, y2 = 3.0, 0.25, 4.0, 0.75
    elif case == 'lower':
        x1, y1, x2, y2 = 3.0, -0.25, 4.0, -0.75
    else:
        raise ValueError("case must be 'center','upper', or 'lower'")

    i1, j1 = xy_to_idx(x1, y1)
    i2, j2 = xy_to_idx(x2, y2)
    mask[min(i1, i2):max(i1, i2), min(j1, j2):max(j1, j2)] = 1.0
    return mask

def create_desired_heightmap(case: str, shape):
    H, W = shape
    hm = np.zeros((H, W), dtype=np.float32)

    def xy_to_idx(x, y):
        i = int((x / 4.0) * (H - 1))
        j = int(((y + 1.0) / 2.0) * (W - 1))
        return i, j

    if case == 'center':
        x1, y1, x2, y2 = 1.5, -0.5, 4.5, 0.5
    elif case == 'upper':
        x1, y1, x2, y2 = 3.0, 0.25, 4.0, 0.75
    elif case == 'lower':
        x1, y1, x2, y2 = 3.0, -0.25, 4.0, -0.75
    else:
        raise ValueError("case must be 'center','upper', or 'lower'")

    i1, j1 = xy_to_idx(x1, y1)
    i2, j2 = xy_to_idx(x2, y2)
    hm[min(i1, i2):max(i1, i2), min(j1, j2):max(j1, j2)] = 0.2
    return hm

def get_random_mid_final_input(result_dir: str):
    """Get a random mid-final input heightmap"""
    mid_final_dirs = [d for d in os.listdir(result_dir) if d.startswith('mid-final') and os.path.isdir(os.path.join(result_dir, d))]
    if not mid_final_dirs:
        raise ValueError("No mid-final directories found")
    
    random_dir = np.random.choice(mid_final_dirs)
    input_path = os.path.join(result_dir, random_dir, 'input.npy')
    if not os.path.exists(input_path):
        raise ValueError(f"Input file not found at {input_path}")
    
    return np.load(input_path)

# === Main ===
if __name__ == "__main__":
    model_path = os.path.join(project_root, "data/ml_models/sequence_v5_time_light.pth")
    result_dir = "/home/harry/AutoGrading/ml_data/0.5"
    
    # Choose scenario
    scenario = 'init-mid'  # or 'mid-final'
    
    if scenario == 'init-mid':
        # Load initial heightmap
        height = 0.5
        init_hmap = np.load(f"/home/harry/hmap/initial/{height}_height.npy")
    else:  # mid-final
        # Get random mid-final input
        init_hmap = get_random_mid_final_input(result_dir)
        print(f"Using mid-final input from {result_dir}")

    case = 'center'
    desired_hmap = create_desired_heightmap(case, shape=(200, 100))

    optimized_ctrl = optimize_action(
        initial_heightmap=init_hmap,
        desired_heightmap=desired_hmap,
        model_path=model_path,
        lr=5e-4,
        num_iters=3000,
        time_weight=5e-4,
        case=case,
        scenario=scenario
    )

    print(f"\nOptimized control pairs (pitch, vertical) for {scenario}:")
    print(optimized_ctrl.reshape(2, 2))

    # save to txt file
    output_path = '/home/harry/ctrl_cmd.txt'
    # make sure it's shaped (2,2)
    ctrl = optimized_ctrl.reshape(2, 2)

    with open(output_path, 'w') as f:
        for i, (pitch, vertical) in enumerate(ctrl):
            # template: i, i+1, 0.0, pitch, vertical, 1.0
            line = f"{pitch},{vertical}\n"
            f.write(line)

    print(f"Wrote control commands to {output_path}")

    optimized_pred, predicted_time = predict_full_output(init_hmap, optimized_ctrl, model_path)

    # === Plot ===
    vmin = min(optimized_pred.min(), desired_hmap.min(), init_hmap.min())
    vmax = max(optimized_pred.max(), desired_hmap.max(), init_hmap.max())
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    # Initial heightmap
    im0 = axs[0].imshow(init_hmap.T, origin='lower', cmap='terrain', extent=[0, 5, -1, 1],
                        vmin=vmin, vmax=vmax)
    axs[0].set_title("Initial Heightmap")
    axs[0].set_xlabel("X"); axs[0].set_ylabel("Y")
    
    # Predicted heightmap
    im1 = axs[1].imshow(optimized_pred.T, origin='lower', cmap='terrain', extent=[0, 5, -1, 1],
                        vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Prediction ({case})\nTime = {predicted_time:.2f}")
    axs[1].set_xlabel("X"); axs[1].set_ylabel("Y")
    
    # Desired heightmap
    im2 = axs[2].imshow(desired_hmap.T, origin='lower', cmap='terrain', extent=[0, 5, -1, 1],
                        vmin=vmin, vmax=vmax)
    axs[2].set_title("Desired")
    axs[2].set_xlabel("X"); axs[2].set_ylabel("Y")
    
    fig.colorbar(im2, ax=axs, location='right', shrink=0.8, label="Height (z)")
    plt.show()
