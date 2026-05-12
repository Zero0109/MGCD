from transformers import AutoFeatureExtractor, SwinModel
import torch
from PIL import Image

extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
model     = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
model.eval()

task = 'pursuit'
scenes = ['hunting', 'finishing', 'collaboration', 'robustness']
type = 'images'
for scene in scenes:
    for i in range(50):
        img    = Image.open(f"/data/terrainEmb/{task}/{type}/{scene}/{i}.png").convert("RGB")
        inputs = extractor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state

        mean_embedding = last_hidden_state.mean(dim=1)
        torch.save(mean_embedding, f'/data/terrainEmb/{task}/represtation/{type}/{scene}/{i}.pt')
