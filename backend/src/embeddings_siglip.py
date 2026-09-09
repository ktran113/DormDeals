"""
Owns the SigLIP model for the served endpoint.

SigLIP replaced CLIP on /search after benchmarking the two on an identical
held-out set: recall@5 0.327 -> 0.690, MRR 0.269 -> 0.594 (see results/).
It costs roughly 4x the compute per image — 196 patches at patch16/224 against
CLIP's 49 — which is why the latency benchmark was re-run after the swap.

embeddings.py still owns CLIP. It is not dead code: generate_embeddings.py
fills the CLIP column with it, and evaluate.py uses it for the photo filter and
near-duplicate guard, which must stay on CLIP so that swapping the retrieval
encoder cannot change which queries are in the eval set.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SiglipVisionModel

MODEL_ID = "google/siglip-base-patch16-224"

#vision tower only — AutoProcessor would pull in SigLIP's SentencePiece
#tokenizer, which is never used here
model = SiglipVisionModel.from_pretrained(MODEL_ID)
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model.eval()  #eval mode disables dropout


def gen_embeddings(img):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    normalized = F.normalize(outputs.pooler_output, p=2, dim=1)
    #squeeze matches embeddings.gen_embeddings: (768,) for one image,
    #(N, 768) for a batch. float32 because FAISS requires it.
    return normalized.squeeze().numpy().astype(np.float32)
