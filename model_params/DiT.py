from easydict import EasyDict


model_params={
    "DiT-XL/2":EasyDict(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16
    ),
    "DiT-XL/4":EasyDict(
        depth=28, hidden_size=1152, patch_size=4, num_heads=16
    ),
    "DiT-XL/8":EasyDict(
        depth=28, hidden_size=1152, patch_size=8, num_heads=16
    ),
    "DiT-L/2":EasyDict(
        depth=24, hidden_size=1024, patch_size=2, num_heads=16
    ),
    "DiT-L/4":EasyDict(
        depth=24, hidden_size=1024, patch_size=4, num_heads=16
    ),
    "DiT-L/8":EasyDict(
        depth=24, hidden_size=1024, patch_size=8, num_heads=16
    ),
    "DiT-B/2":EasyDict(
        depth=12, hidden_size=768, patch_size=2, num_heads=12
    ),
    "DiT-B/4":EasyDict(
        depth=12, hidden_size=768, patch_size=4, num_heads=12
    ),
    "DiT-B/8":EasyDict(
        depth=12, hidden_size=768, patch_size=8, num_heads=12
    ),
    "DiT-S/2":EasyDict(
        depth=12, hidden_size=384, patch_size=2, num_heads=6
    ),
    "DiT-S/4":EasyDict(
        depth=12, hidden_size=384, patch_size=4, num_heads=6
    ),
    "DiT-S/8":EasyDict(
        depth=12, hidden_size=384, patch_size=8, num_heads=6
    )

}