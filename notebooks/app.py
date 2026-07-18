import torch
import torch.nn as nn
from torchvision import transforms
import timm
import gradio as gr

# ============================================
# 1. Device
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# 2. Class Names
# ============================================
CLASS_NAMES = ['Barbari', 'Goat', 'Harri', 'Naeimi', 'Najdi', 'Roman', 'Sawakni']
NUM_CLASSES = len(CLASS_NAMES)

# ============================================
# 3. CBAM Module
# ============================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))) * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg   = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1))) * x

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    def forward(self, x):
        return self.spatial_att(self.channel_att(x))

# ============================================
# 4. Model
# ============================================
class SwinWithIntermediateCBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.swin  = timm.create_model('swin_tiny_patch4_window7_224',
                                        pretrained=False,
                                        num_classes=0,
                                        global_pool='')
        self.cbam1 = CBAM(96)
        self.cbam2 = CBAM(192)
        self.cbam3 = CBAM(384)
        self.head  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def apply_cbam(self, x, cbam, H, W):
        if x.dim() == 4:
            B, H2, W2, C = x.shape
            x = x.permute(0, 3, 1, 2)
            x = cbam(x)
            x = x.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            B, L, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = cbam(x)
            x = x.reshape(B, C, H*W).permute(0, 2, 1)
        return x

    def forward(self, x):
        x = self.swin.patch_embed(x)
        for i, layer in enumerate(self.swin.layers):
            x = layer(x)
            if i == 0:   x = self.apply_cbam(x, self.cbam1, 28, 28)
            elif i == 1: x = self.apply_cbam(x, self.cbam2, 14, 14)
            elif i == 2: x = self.apply_cbam(x, self.cbam3, 7, 7)
        x = self.swin.norm(x)
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, H*W, C)
        return self.head(x.permute(0, 2, 1))

# ============================================
# 5. Load Model
# ============================================
model = SwinWithIntermediateCBAM(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print('Model loaded!')

# ============================================
# 6. Transform
# ============================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================
# 7. Predict Function
# ============================================
def predict(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)[0]
    top3_probs, top3_idx = torch.topk(probs, 3)
    return {CLASS_NAMES[i]: round(p.item()*100, 2)
            for p, i in zip(top3_probs, top3_idx)}

# ============================================
# 8. Gradio App
# ============================================
custom_css = """
    footer { display: none !important; }
    .label-output .output-class {
        font-size: 52px !important;
        font-weight: 900 !important;
        text-align: center !important;
        padding: 20px !important;
    }
    .label-output .label-name { font-size: 26px !important; font-weight: 700 !important; }
    .label-output .label-confidence { font-size: 26px !important; font-weight: 700 !important; }
    .label-output { min-height: 350px !important; padding: 20px !important; }
    h1 { font-size: 42px !important; text-align: center !important; }
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Upload Sheep Image', sources=['upload']),
    outputs=gr.Label(num_top_classes=3, label='Breed Prediction'),
    title='🐑 Sheep Breed Recognition System',
    description=None,
    theme=gr.themes.Soft(),
    css=custom_css
)

demo.launch()
