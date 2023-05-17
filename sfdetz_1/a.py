utdc_classes = ('echinus','starfish','holothurian','scallop')
templates = [
    'a photo of a {}.',
    'a photo of a small {}.',
    'a photo of a medium {}.',
    'a photo of a large {}.',
    'This is a photo of a {}.',
    'This is a photo of a small {}.',
    'This is a photo of a medium {}.',
    'This is a photo of a large {}.',
    'a {} in the scene.',
    'a photo of a {} in the scene.',
    'There is a {} in the scene.',
    'There is the {} in the scene.',
    'This is a {} in the scene.',
    'This is the {} in the scene.',
    'This is one {} in the scene.',
    ]
import torch
import numpy as np
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)

## single template
def single_templete(save_path, class_names, model):
    with torch.no_grad():
        texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).cuda()
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        np.save(save_path, text_embeddings.cpu().numpy())
    return text_embeddings

## multi templates
def multi_templete(save_path, class_names, model, templates):
    with torch.no_grad():
        text_embeddings = []
        for classname in class_names:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_embeddings.append(class_embedding)
        text_embeddings = torch.stack(text_embeddings, dim=0).cuda()
    np.save(save_path, text_embeddings.cpu().numpy())
    return text_embeddings

save_path='utdac_multi.npy'
text_embeddings = multi_templete(save_path, COCO_classes, model, templates)