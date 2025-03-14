import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from mmseg.ops import resize
from omegaconf import OmegaConf

from models.builder import MODELS, build_model

DINO_NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@MODELS.register_module()
class LPOSS(nn.Module):
    def __init__(self, clip_backbone, class_names, vit_arch="vit_base", vit_patch_size=16, enc_type_feats="k"):
        super(LPOSS, self).__init__()

        # ==== build MaskCLIP backbone =====
        maskclip_cfg = OmegaConf.load(f"configs/{clip_backbone}.yaml")
        self.clip_backbone = build_model(maskclip_cfg["model"], class_names=class_names)
        for param in self.clip_backbone.parameters():
            param.requires_grad = False

        # ==== build DINO backbone =====
        self.dino_T = DINO_NORMALIZE
        self.dino_arch = vit_arch
        self.enc_type_feats = enc_type_feats
        self.dino_patch_size = vit_patch_size
        if self.dino_arch == "vit_base":
            self.dino_encoder = torch.hub.load('facebookresearch/dino:main', f'dino_vitb{self.dino_patch_size}')
        elif self.dino_arch == "vit_small":
            self.dino_encoder = torch.hub.load('facebookresearch/dino:main', f'dino_vits{self.dino_patch_size}')
        self.hook_features = {}
        def hook_fn_forward_qkv(module, input, output):
            self.hook_features["qkv"] = output

        self.dino_encoder._modules["blocks"][-1]._modules["attn"]._modules[
            "qkv"
        ].register_forward_hook(hook_fn_forward_qkv)


    def make_input_divisible(self, x: torch.Tensor, patch_size) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (patch_size - W_0 % patch_size) % patch_size
        pad_h = (patch_size - H_0 % patch_size) % patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x
    

    @torch.no_grad()
    def extract_feats(self, type_feats="v"):
        """
        DINO feature extractor. Attaches a hook on the last attention layer.
        :param type_feats: (string) - type of features from DINO ViT
        """
        nh = self.dino_encoder.blocks[-1].attn.num_heads
        nb_im, nb_tokens, C_qkv = self.hook_features["qkv"].shape

        qkv = (
            self.hook_features["qkv"]
                .reshape(
                nb_im, nb_tokens, 3, nh, C_qkv // nh // 3
            )  # 3 corresponding to |qkv|
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if type_feats == "q":
            return q.transpose(1, 2).float()
        elif type_feats == "k":
            return k.transpose(1, 2).float()
        elif type_feats == "v":
            return v.transpose(1, 2).float()
        else:
            raise ValueError("Unknown features")
    

    @torch.no_grad()
    def get_dino_features(self, x: torch.Tensor):
        """
        Extracts dense DINO features.

        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - of dense DINO features, (int, int) - size of feature map
        """
        x = self.make_input_divisible(x, self.dino_patch_size)
        batch = self.dino_T(x)  # tensor B C H W
        h_featmap = batch.shape[-2] // self.dino_patch_size
        w_featmap = batch.shape[-1] // self.dino_patch_size

        # Forward pass
        # Encoder forward pass and get hooked intermediate values
        _ = self.dino_encoder(batch)

        # Get decoder features
        feats = self.extract_feats(type_feats=self.enc_type_feats)
        num_extra_tokens = 1

        # B nbtokens+1 nh dim
        feats = feats[:, num_extra_tokens:, :, :].flatten(-2, -1).permute(0, 2, 1)  # B C nbtokens
        # B, C, nbtokens
        feats = feats / feats.norm(dim=1, keepdim=True)  # normalize features

        return feats, (h_featmap, w_featmap)


    @torch.no_grad()
    def get_clip_features(self, x: torch.Tensor):
        """
        Extracts MaskCLIP features
        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - clip dense features, (torch.Tensor) - output probabilities
        """
        x = self.make_input_divisible(x, self.clip_backbone.patch_size)
        maskclip_map, feat = self.clip_backbone(x, return_feat=True)

        return feat, maskclip_map


    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        dino_feats, (h_dino, w_dino) = self.get_dino_features(x)
        clip_feats, clip_preds = self.get_clip_features(x)
        h_clip, w_clip = clip_feats.shape[-2:]
        clf = F.normalize(self.clip_backbone.decode_head.class_embeddings, p=2, dim=-1)

        num_classes = len(self.clip_backbone.decode_head.class_names) # clip_preds.shape[1]

        dino_feats = reshape_features(dino_feats)
        dino_feats = F.normalize(dino_feats, p=2, dim=-1)

        clip_feats = reshape_features(clip_feats)
        clip_feats = F.normalize(clip_feats, p=2, dim=-1)

        clip_feats = clip_feats.reshape((clip_feats.shape[0], h_clip, w_clip, -1))
        dino_feats = dino_feats.reshape((dino_feats.shape[0], h_dino, w_dino, -1))

        # clip_feats = torch.unsqueeze(clip_feats, 0)
        # dino_feats = torch.unsqueeze(dino_feats, 0)

        return dino_feats, clip_feats, clf
    

def reshape_features(feats):
    # feats = feats[0, ...]
    feats = feats.reshape((feats.shape[0], feats.shape[1], -1))
    feats = feats.permute(0, 2, 1)

    return feats