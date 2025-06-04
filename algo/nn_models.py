import torch.nn as nn
import torch


# --- Neural Network Definition ---
class ForwardCNN(nn.Module):
    def __init__(self, action_dim=4):  # Changed to 4 for 2x2 action vector
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=6, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=6, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=6, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((10,4)),
            nn.Flatten()
        )
        self.feature_dim = 128 * 10 * 4

        # Action branch
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, 256), nn.ReLU(),
            nn.Linear(256, self.feature_dim), nn.ReLU()
        )

        # Fusion MLP
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 2048), nn.ReLU(),
            nn.Linear(2048, self.feature_dim), nn.ReLU()
        )

        # Time head
        self.time_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(200,100), mode='bilinear', align_corners=False)
        )

    def forward(self, heightmap, actions):
        bs = heightmap.size(0)
        h_feat = self.encoder(heightmap)
        a_feat = self.action_fc(actions)
        fused = torch.cat([h_feat, a_feat], dim=1)
        shared = self.fc(fused)

        # Time prediction
        pred_time = self.time_head(shared)

        # Heightmap prediction
        x = shared.view(bs, 128, 10, 4)
        pred_hmap = self.decoder(x)

        return pred_hmap, pred_time

# --- Lightweight Neural Network Definition ---
class LightForwardCNN(nn.Module):
    def __init__(self, action_dim=4):
        super().__init__()
        # Encoder - reduced filters and kernel sizes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,4)),
            nn.Flatten()
        )
        self.feature_dim = 32 * 8 * 4
        
        # Action branch - reduced hidden dimensions
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, 64), nn.ReLU(),
            nn.Linear(64, self.feature_dim), nn.ReLU()
        )
        
        # Fusion MLP - smaller network
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, self.feature_dim), nn.ReLU()
        )
        
        # Time head - fewer layers, no batch norm
        self.time_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Decoder - fewer filters
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(200,100), mode='bilinear', align_corners=False)
        )
    
    def forward(self, heightmap, actions):
        bs = heightmap.size(0)
        h_feat = self.encoder(heightmap)
        a_feat = self.action_fc(actions)
        fused = torch.cat([h_feat, a_feat], dim=1)
        shared = self.fc(fused)
        
        # Time prediction
        pred_time = self.time_head(shared)
        
        # Heightmap prediction
        x = shared.view(bs, 32, 8, 4)
        pred_hmap = self.decoder(x)
        
        return pred_hmap, pred_time

# --- Tiny Neural Network Definition ---
class TinyForwardCNN(nn.Module):
    def __init__(self, action_dim=4):
        super().__init__()
        # Minimal encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,2)),
            nn.Flatten()
        )
        self.feature_dim = 16 * 4 * 2  # 128 features
        
        # Minimal action branch
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, 32), nn.ReLU(),
            nn.Linear(32, 128)
        )
        
        # Minimal fusion layer
        self.fc = nn.Linear(self.feature_dim * 2, 128)
        
        # Simple time head
        self.time_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Minimal decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(200,100), mode='bilinear', align_corners=False)
        )
    
    def forward(self, heightmap, actions):
        bs = heightmap.size(0)
        h_feat = self.encoder(heightmap)
        a_feat = self.action_fc(actions)
        fused = torch.cat([h_feat, a_feat], dim=1)
        shared = self.fc(fused)
        
        # Time prediction
        pred_time = self.time_head(shared)
        
        # Heightmap prediction
        x = shared.view(bs, 16, 4, 2)
        pred_hmap = self.decoder(x)
        
        return pred_hmap, pred_time