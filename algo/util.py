import os
import numpy as np
import matplotlib.pyplot as plt

class PointCloudLoader:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.point_clouds = {}

    def load_all(self, limit=None):
        count = 0
        for subdir in os.listdir(self.root_folder):
            if limit is not None and count >= limit:
                break
            subfolder_path = os.path.join(self.root_folder, subdir)
            if os.path.isdir(subfolder_path):
                csv_path = os.path.join(subfolder_path, "fluid0.csv")
                if os.path.exists(csv_path):
                    try:
                        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                        # Apply filters: z > 0 and 0 < x < 5 and -1.0 < y < 1.0
                        filtered_data = data[(data[:, 2] > 0.0)  & (data[:, 0] > 0.0) & (data[:, 0] < 5.0) & (data[:, 1] > -1.0) & (data[:, 1] < 1.0) ]
                        self.point_clouds[subdir] = filtered_data
                        print(f"Loaded: {csv_path} | Points after filter: {len(filtered_data)}")
                        count += 1
                    except Exception as e:
                        print(f"Error loading {csv_path}: {e}")
                else:
                    print(f"fluid0.csv not found in {subfolder_path}")
        if limit is not None:
            print(f"\n✅ Loaded {count}/{limit} point clouds (limit applied).")
        else:
            print(f"\n✅ Loaded {count} point clouds.")

    def get_data(self):
        return self.point_clouds

    def pointcloud_to_heightmap(self, subfolder_name, grid_size=(200, 100)):
        if subfolder_name not in self.point_clouds:
            print(f"No data found for subfolder: {subfolder_name}")
            return None

        data = self.point_clouds[subfolder_name]
        if data.size == 0:
            print(f"No points to process for {subfolder_name}.")
            return None

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        xmin, xmax = 0.0, 5.0
        ymin, ymax = -1.0, 1.0

        # Bin data into grid
        x_bins = np.linspace(xmin, xmax, grid_size[0] + 1)
        y_bins = np.linspace(ymin, ymax, grid_size[1] + 1)

        heightmap = np.full((grid_size[0], grid_size[1]), 0.0)  # Initialize with zeros

        # For each bin, compute max z
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                mask = (x >= x_bins[i]) & (x < x_bins[i + 1]) & \
                       (y >= y_bins[j]) & (y < y_bins[j + 1])
                if np.any(mask):
                    heightmap[i, j] = np.max(z[mask])  # max height

        return heightmap

    def save_all_heightmaps(self, save_folder, grid_size=(200, 100)):
        os.makedirs(save_folder, exist_ok=True)
        count = 0
        for subfolder_name in self.point_clouds.keys():
            if not subfolder_name.startswith("soil_leveling_"):
                continue
            try:
                idx = int(subfolder_name.replace("soil_leveling_", ""))
            except ValueError:
                print(f"Skipping {subfolder_name} (invalid index)")
                continue

            heightmap = self.pointcloud_to_heightmap(subfolder_name, grid_size)
            if heightmap is not None:
                save_path = os.path.join(save_folder, f"init0.37_{idx}.npy")
                np.save(save_path, heightmap)
                print(f"Saved heightmap to: {save_path}")
                count += 1
        print(f"\n✅ Saved {count} heightmaps to {save_folder}")


