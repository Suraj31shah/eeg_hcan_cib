from collections import OrderedDict
import torch
import torch.nn as nn


# Official AMIGOS 14-channel Emotiv EPOC order:
# 0:AF3, 1:F7, 2:F3, 3:FC5, 4:T7, 5:P7, 6:O1, 7:O2, 8:P8, 9:T8, 10:FC6, 11:F4, 12:F8, 13:AF4
AMIGOS_REGIONS = OrderedDict([
    ("frontal", [0, 1, 2, 3, 10, 11, 12, 13]),
    ("temporal", [4, 9]),
    ("parietal", [5, 8]),
    ("occipital", [6, 7]),
])


class RegionGroupedFusion(nn.Module):
    def __init__(self, d_embed: int = 64, n_classes: int = 2, region_dropout_p: float = 0.2):
        super().__init__()
        self.region_dropout_p = float(region_dropout_p)
        self.regions = OrderedDict(AMIGOS_REGIONS)
        self.region_names = list(self.regions.keys())

        self.intra_attentions = nn.ModuleDict({
            region: nn.Sequential(
                nn.Linear(d_embed, max(8, d_embed // 2)),
                nn.Tanh(),
                nn.Linear(max(8, d_embed // 2), 1),
                nn.Softmax(dim=1),
            )
            for region in self.regions
        })

        self.group_classifiers = nn.ModuleDict({
            region: nn.Linear(d_embed, n_classes) for region in self.regions
        })

        self.final_classifier = nn.Linear(len(self.regions) * d_embed, n_classes)

    def extract_region_tokens(self, x_embed: torch.Tensor):
        """
        x_embed: `[B, 14, d_embed]`
        Returns:
            group_feats: list of 4 `[B, d_embed]`
            group_logits: list of 4 `[B, n_classes]`
        """
        B = x_embed.size(0)
        n_regions = len(self.regions)
        if self.training and self.region_dropout_p > 0:
            mask = torch.empty(B, n_regions, device=x_embed.device).bernoulli_(1.0 - self.region_dropout_p)
            mask = mask / (1.0 - self.region_dropout_p)
        else:
            mask = torch.ones(B, n_regions, device=x_embed.device)

        group_feats = []
        group_logits = []
        for idx, (region, channels) in enumerate(self.regions.items()):
            region_x = x_embed[:, channels, :]
            attn = self.intra_attentions[region](region_x)
            pooled = torch.sum(region_x * attn, dim=1)
            pooled = pooled * mask[:, idx : idx + 1]
            group_feats.append(pooled)
            group_logits.append(self.group_classifiers[region](pooled))
        return group_feats, group_logits

    def forward(self, x_embed: torch.Tensor):
        group_feats, group_logits = self.extract_region_tokens(x_embed)
        fused = torch.cat(group_feats, dim=1)
        final_logits = self.final_classifier(fused)
        return final_logits, group_logits

