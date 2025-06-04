import os
import subprocess
import sys
import numpy as np
import time
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from algo.algo_seq_leveltwice import optimize_action, create_desired_heightmap, predict_full_output
from algo.util import PointCloudLoader

def run_optimization(pile_height: float, output_dir: str, heightmap_path: str, push_seq: str):
    """Run optimization and save results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load heightmap
    init_hmap = np.load(heightmap_path)
    
    # Create desired heightmap
    desired_hmap = create_desired_heightmap('center', shape=(200, 100))
    
    # Run optimization
    model_path = os.path.join(project_root, "data/ml_models/sequence_v5_time_light.pth")
    optimized_ctrl = optimize_action(
        initial_heightmap=init_hmap,
        desired_heightmap=desired_hmap,
        model_path=model_path,
        lr=5e-4,
        num_iters=2000,
        time_weight=1e-6,
        case='center',
        scenario='init-mid'
    )
    
    # Save control commands
    output_path = os.path.join(output_dir, f"{pile_height:.6f}_{push_seq}.txt")
    ctrl = optimized_ctrl.reshape(2, 2)
    
    with open(output_path, 'w') as f:
        for pitch, vertical in ctrl:
            line = f"{pitch},{vertical}\n"
            f.write(line)
    
    print(f"Saved control commands to {output_path}")
    return output_path

def run_simulation(pile_height: float, push_seq: str):
    """Run the Chrono simulation"""
    demo_path = "/home/harry/chrono-wisc/build/bin/demo_VEH_CRMTerrain_LevelValidation"
    
    # Default vehicle initial state
    veh_init_state = "-2.0,0.0,0.3,1.0,0.0,0.0,0.0"
    
    # Construct command
    cmd = [
        demo_path,
        "--pile_height", str(pile_height),
        "--push_seq", push_seq,
        "--veh_init_state", veh_init_state
    ]
    
    print("Running simulation with command:", " ".join(cmd))
    
    # Change to demo directory before running
    demo_dir = "/home/harry/chrono-wisc/build"
    original_dir = os.getcwd()
    try:
        os.chdir(demo_dir)
        subprocess.run(cmd)
    finally:
        os.chdir(original_dir)  # Change back to original directory

def wait_for_file(file_path, max_wait_time=600, check_interval=5):
    """Wait for a file to appear, with timeout"""
    file_path = Path(file_path)
    start_time = time.time()
    
    print(f"Waiting for file: {file_path}")
    while not file_path.exists():
        if time.time() - start_time > max_wait_time:
            print(f"Timeout: File {file_path} did not appear within {max_wait_time} seconds")
            return False
        time.sleep(check_interval)
        print(f"Still waiting... ({int(time.time() - start_time)} seconds elapsed)")
    
    print(f"File found after {int(time.time() - start_time)} seconds")
    return True

def convert_pointcloud_to_heightmap(pile_height, push_seq):
    """Convert the simulation's pointcloud to a heightmap"""
    # Paths
    output_dir = project_root+"/data/output/"+f"{pile_height:.6f}"
    print(f"output_dir: {output_dir}")
    pc_dir = os.path.join(output_dir, f"soil_leveling_{push_seq}")
    # Ensure pointcloud directory exists
    os.makedirs(pc_dir, exist_ok=True)
    save_dir = project_root+"/data/temp"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create pointcloud loader
    loader = PointCloudLoader(output_dir)
    
    # Load only the specific folder
    folder_name = f"soil_leveling_{push_seq}"
    
    # Wait for fluid*.csv file to appear (max 2000 seconds)
    start_time = time.time()
    fluid_files = []
    while time.time() - start_time < 2000:  # 2000 seconds timeout
        fluid_files = [f for f in os.listdir(pc_dir) if f.startswith('fluid') and f.endswith('.csv')]
        if fluid_files:
            break
        print(f"Waiting for fluid*.csv file... ({int(time.time() - start_time)} seconds elapsed)")
        time.sleep(5)  # Check every 5 seconds
    
    if not fluid_files:
        raise FileNotFoundError(f"No fluid*.csv file found in {pc_dir} after 2000 seconds")
    
    if len(fluid_files) > 1:
        print(f"Warning: Multiple fluid files found in {pc_dir}, using the first one: {fluid_files[0]}")
    
    csv_path = os.path.join(pc_dir, fluid_files[0])
    
    # Give the file a little time to be fully written
    time.sleep(2)
    
    # Load the point cloud
    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        # Apply filters: z > 0 and 0 < x < 5 and -1.0 < y < 1.0
        filtered_data = data[(data[:, 2] > 0.0) & (data[:, 0] > 0.0) & (data[:, 0] < 5.0) & (data[:, 1] > -1.0) & (data[:, 1] < 1.0)]
        loader.point_clouds[folder_name] = filtered_data
        print(f"Loaded: {csv_path} | Points after filter: {len(filtered_data)}")
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        raise
    
    # Convert to heightmap
    heightmap = loader.pointcloud_to_heightmap(folder_name)
    if heightmap is None:
        raise ValueError(f"Failed to convert pointcloud to heightmap for {folder_name}")
    
    # Save heightmap
    heightmap_path = os.path.join(save_dir, f"{pile_height:.6f}_{push_seq}_heightmap.npy")
    np.save(heightmap_path, heightmap)
    print(f"Saved heightmap to: {heightmap_path}")
    
    return heightmap_path

def main():
    # Parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pile_height', type=float, default=0.37,
                      help='Height of the pile (default: 0.37)')
    args = parser.parse_args()
    pile_height = args.pile_height
    output_dir = os.path.join(project_root, "data", "control_commands")
    
    # Step 1: Run first optimization with initial heightmap
    initial_heightmap_path = project_root+f"/data/init_heightmaps/{pile_height}_height.npy"
    run_optimization(pile_height, output_dir, initial_heightmap_path, "firstpush")
    
    # Step 2: Run first simulation (firstpush)
    #run_simulation(pile_height, "firstpush")
    
    # Step 3: Wait for and process the pointcloud from first simulation
    try:
        mid_heightmap_path = convert_pointcloud_to_heightmap(pile_height, "firstpush")
        
        # Step 4: Run second optimization with mid heightmap
        run_optimization(pile_height, output_dir, mid_heightmap_path, "secondpush")
        
        # # Step 5: Run second simulation (secondpush)
        # run_simulation(pile_height, "secondpush")
    except Exception as e:
        print(f"Error processing second part of simulation: {e}")

if __name__ == "__main__":
    main()
