"""ViT-B/16 deepfake detector.

Previously this constructed `torchvision.models.VisionTransformer` directly with
`image_size=255, patch_size=51`, which meant:

* **`pretrained` was accepted and ignored.** Two models built with `pretrained=True` and
  `pretrained=False` came out with bit-identical weights, and no ViT checkpoint was ever
  fetched -- the torch.hub cache holds b4, resnest50, squeezenet and swin_t, and nothing
  else. So every run trained 91M parameters from scratch, which on 2000 images over three
  epochs at lr 1e-4 lands exactly where you would expect: AUROC 0.50-0.56, at chance, in
  both the natural and the balanced sweep.
* **`finetune` was accepted and ignored.** The freezing block was commented out, so
  `--finetune` left all 91,070,977 parameters trainable.
* A 255/51 geometry cannot load any published checkpoint anyway: torchvision's ViT-B/16
  weights require 224/16.

Now it wraps the real `vit_b_16`. Its hidden dimension is 768 and it has 12 layers and 12
heads -- identical to what the hand-rolled version specified -- so the penultimate space
stays `vit_cls768` and `models/uncertainty/capabilities.py` remains accurate.

The pipeline resizes images to 255x255 (`CNNModel.common_transforms`) while torchvision's ViT
requires exactly its configured `image_size`. Rather than change the shared transform for one
detector, the input is interpolated to 224 in `forward`.
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#: What `vit_b_16` requires. Bilinear interpolation from the pipeline's 255x255 costs a
#: fraction of a millisecond per image and keeps the pretrained patch embedding valid.
INPUT_SIZE = 224

#: ViT-B/16's transformer width -- the penultimate (CLS) feature dimension the capability
#: table records as `vit_cls768`.
HIDDEN_DIM = 768


class ModelOut(nn.Module):
    def __init__(self, pretrained: bool = False, finetune: bool = False,
                 exclude_top: bool = False, output_classes: int = 3,
                 classification_strategy: str = 'categorical',
                 configuration: str = 'default'):
        super(ModelOut, self).__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.vit_b_16(weights=weights)

        self.in_features = HIDDEN_DIM
        self.out_features = output_classes if classification_strategy == 'categorical' else 1

        # Freeze before replacing the head, so the new head is always trainable. This is
        # what the other detectors mean by "finetune": a linear probe on frozen features.
        if finetune:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        if exclude_top:
            self.model.heads = Identity()
        else:
            # Deliberately a single Linear at `heads.head`, matching what the previous
            # implementation produced. `models/uncertainty/capabilities.py` records this
            # detector's graft point as `heads.head` with 768 in_features and its
            # penultimate space as `vit_cls768`, and `reconcile()` cross-checks that table
            # against a runtime probe. Widening this into a 1024-d Sequential -- as
            # effnetdf and resnestdf have -- would move the graft point, change the
            # penultimate space, and make the detector newly comparable with those three
            # under `comparable_detector_groups`. That may well be worth doing, but it is a
            # benchmark-design change rather than a bug fix, so it is left alone here.
            self.model.heads = nn.Sequential()
            self.model.heads.add_module(
                "head", nn.Linear(self.in_features, self.out_features)
            )
            if classification_strategy == 'categorical':
                self.model.heads.add_module(
                    module=nn.Softmax(dim=1), name="Categorical_Softmax",
                )

    def forward(self, x):
        # torchvision's ViT asserts the input matches its configured image_size exactly,
        # and the shared transform produces 255x255 for every detector.
        if x.shape[-1] != INPUT_SIZE or x.shape[-2] != INPUT_SIZE:
            x = F.interpolate(
                x, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False,
            )
        return self.model(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def main():
    from torchsummary import summary
    model = ModelOut(False, False)
    summary(model.cuda(), (3, 255, 255))


if __name__ == "__main__":
    main()
