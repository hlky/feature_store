# The Feature Store



## Examples

```python
class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3)

    def forward(self, x):
        return self.conv(x)


class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([A() for _ in range(5)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Main(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = A()
        self.b = B()
        self.layers = nn.ModuleList([B() for _ in range(3)])

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        for layer in self.layers:
            x = layer(x)
        return x


feature_store = FeatureStore()

main = Main().eval()

feature_store.store(
    main,
    feature_wishlist=[
        FeatureStoreItem(
            pattern="",
            config=FeatureConfig(
                type=FeatureType.INPUT,
                pattern_type=None,
                actions=[ActionShapes(enabled=True)],
            ),
        ),
    ],
)
input_tensor = torch.randn(1, 64, 64, 64)
output = main(input_tensor)
```

```python
import torch
from diffusers import UNet2DConditionModel

model = (
    UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", variant="fp16"
    )
    .eval()
    .half()
    .cuda()
)
feature_store.store(
    model,
    feature_wishlist=[
        FeatureStoreItem(
            pattern=r".+.\d+.attentions.\d+.transformer_blocks.\d+.attn1$",
            config=FeatureConfig(
                type=FeatureType.INPUT,
                pattern_type=FeaturePatternType.REGEX,
                actions=[ActionStore(enabled=True)],
            ),
        ),
    ],
)
sample = torch.randn(1, 4, 64, 64).half().cuda()
timestep = 999.0
encoder_hidden_states = torch.randn(1, 77, 768).half().cuda()
model(sample, timestep, encoder_hidden_states)

feature_store_2 = FeatureStore(features=feature_store.features)

model_2 = (
    UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", variant="fp16"
    )
    .half()
    .cuda()
)
feature_store.store(
    model_2,
    feature_wishlist=[
        FeatureStoreItem(
            pattern=r".+.\d+.attentions.\d+.transformer_blocks.\d+.attn1$",
            config=FeatureConfig(
                type=FeatureType.INPUT,
                pattern_type=FeaturePatternType.REGEX,
                actions=[
                    ActionConcat(
                        order=ActionOrder.A_B, dim=1, batch=ActionConcatType.ZEROS
                    )
                ],
            ),
        ),
        FeatureStoreItem(
            pattern=r".+.\d+.attentions.\d+.transformer_blocks.\d+.attn1.to_out.0$",
            config=FeatureConfig(
                type=FeatureType.INPUT,
                pattern_type=FeaturePatternType.REGEX,
                actions=[ActionChunk(chunks=2, dim=1, index=0)],
            ),
        ),
    ],
)
sample_2 = torch.randn(3, 4, 64, 64).half().cuda()
timestep_2 = 999.0
encoder_hidden_states_2 = torch.randn(3, 77, 768).half().cuda()
with torch.inference_mode():
    model_2(sample_2, timestep_2, encoder_hidden_states_2)
```
