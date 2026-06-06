"""Vision-encoder presets used as VLA perception front-ends.

Dimensions follow the public HF configs of the corresponding ViT towers. Used
via ModelAnalyzer(model_id, hw, "configs/vit.py", source="vision_encoders").
num_patches is image-size-driven and chosen here to divide evenly:
  CLIP  ViT-L/14 @336 -> (336/14)^2 = 576 (+1 CLS)
  DINOv2 ViT-L/14 @518 -> (518/14)^2 = 1369 (+1 CLS)
  SigLIP ViT-L/16 @384 -> (384/16)^2 = 576 (no CLS token)
"""

from easydict import EasyDict


model_params = {
    # OpenAI CLIP ViT-Large/14, 336px (LLaVA's vision tower).
    "clip_vit_large_p14_336": EasyDict(
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=4096,
        image_size=336,
        patch_size=14,
        num_channels=3,
        has_cls_token=True,
    ),
    # DINOv2 ViT-Large/14, 518px (one of OpenVLA's two encoders).
    "dinov2_vit_large_p14_518": EasyDict(
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=4096,
        image_size=518,
        patch_size=14,
        num_channels=3,
        has_cls_token=True,
    ),
    # SigLIP ViT-Large/16, 384px (OpenVLA's other encoder; no CLS token).
    "siglip_vit_large_p16_384": EasyDict(
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=4096,
        image_size=384,
        patch_size=16,
        num_channels=3,
        has_cls_token=False,
    ),
    # SigLIP-So400m/14, 224px -- PaliGemma's vision tower (pi0). 16x16 = 256
    # patch tokens, no CLS token.
    "siglip_so400m_p14_224": EasyDict(
        hidden_size=1152,
        num_attention_heads=16,
        num_hidden_layers=27,
        intermediate_size=4304,
        image_size=224,
        patch_size=14,
        num_channels=3,
        has_cls_token=False,
    ),
}
