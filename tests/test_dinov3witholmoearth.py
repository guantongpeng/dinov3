import sys
sys.path.insert(0, '/home/guantp/pro/dinov3')

import torch
from functools import partial
from dinov3.data.loaders import make_dataset
from dinov3.data.augmentations import DataAugmentationDINO
from dinov3.data.collate import collate_h5_olmoearth_and_cast
from dinov3.data.masking import MaskingGenerator
from dinov3.train.olmoearth_inference import load_olmoearth_model, run_olmoearth_inference

# Build dataset and dataloader
aug = DataAugmentationDINO(
    global_crops_scale=(0.4, 1.0),
    local_crops_scale=(0.05, 0.4),
    local_crops_number=8,
    global_crops_size=224,
    local_crops_size=96,
)
dataset = make_dataset(
    dataset_str='H5OlmoEarth:root=/tmp/test_h5_olmoearth_v2',
    transform=aug,
    target_transform=lambda _: (),
)
print(f'Dataset: {len(dataset)} samples')

# Build collate
img_size = 224
patch_size = 16
n_tokens = (img_size // patch_size) ** 2
mask_gen = MaskingGenerator(
    input_size=(img_size // patch_size, img_size // patch_size),
    max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
)
collate_fn = partial(
    collate_h5_olmoearth_and_cast,
    mask_ratio_tuple=(0.1, 0.5),
    mask_probability=0.5,
    dtype=torch.float32,
    n_tokens=n_tokens,
    mask_generator=mask_gen,
)

# Get a batch
samples = [dataset[i % len(dataset)] for i in range(2)]
batch = collate_fn(samples)

print(f'Batch keys: {list(batch.keys())}')
print(f"  collated_global_crops: {batch['collated_global_crops'].shape}")
print(f"  collated_local_crops: {batch['collated_local_crops'].shape}")
print(f'  olmoearth_modalities:')
for k, v in batch['olmoearth_modalities'].items():
    print(f'    {k}: {v.shape}')

# Run OLMoEarth inference
print()
print('Loading OLMoEarth model...')
model = load_olmoearth_model('OlmoEarth-v1-Nano', device=torch.device('cuda'))

print('Running OLMoEarth inference...')
embeddings = run_olmoearth_inference(
    model,
    batch['olmoearth_modalities'],
    batch['olmoearth_metadata'],
    patch_size=4,
    device=torch.device('cuda'),
)

for mod_name, emb in embeddings.items():
    print(f'  {mod_name}: shape={emb.shape}')

# Verify frozen
for p in model.parameters():
    assert not p.requires_grad
print('OLMoEarth model params frozen: OK')

print()
print('Full pipeline test PASSED!')