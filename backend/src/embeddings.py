"""
Owns the CLIP model. Import gen_embeddings from here (server and scripts).
Loading this module loads CLIP into memory.
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()  #eval mode disables dropout


def gen_embeddings(img):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():  #no grad less memory
        outputs = model.get_image_features(**inputs)
        embedding = outputs.pooler_output
    normalized = F.normalize(embedding, p=2, dim=1)
    return normalized.squeeze().numpy()
