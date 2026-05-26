import torch

# PyTorch 2.6 defaults weights_only=True, but mmengine checkpoints contain
# non-standard globals (HistoryBuffer, numpy._reconstruct, etc.).
# Monkey-patch torch.load to restore the old default for trusted checkpoints.
_original_load = torch.load
torch.load = lambda *a, **kw: _original_load(*a, **{'weights_only': False, **kw})

from mmdet.apis import DetInferencer

inferencer = DetInferencer(
    model='rtmdet_tiny_8xb32-300e_coco.py',
    weights='rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
    device='cuda:0',
)
inferencer('demo/demo.jpg', out_dir='output', show=False)

