# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
import math
import os
from typing import List

from einops import rearrange
from pose_tracking.models.cnnlstm import MLP
from pose_tracking.models.detr import CNNFeatureExtractor
from pose_tracking.utils.detr_utils import get_crops
from pose_tracking.utils.misc import print_cls
from pose_tracking.utils.vis import box_cxcywh_to_xyxy
import torch
import torch.nn as nn
import torch.nn.functional as F
from memotr.structures.track_instances import TrackInstances
from memotr.utils.nested_tensor import NestedTensor
from memotr.utils.utils import inverse_sigmoid
from torch.utils.checkpoint import checkpoint

from .backbone import BackboneWithPE
from .backbone import build as build_backbone_with_pe
from .deformable_transformer import DeformableTransformer
from .deformable_transformer import build as build_deformable_transformer
from .ffn import FFN
# from .mlp import MLP
from .query_updater import build as build_query_updater, QueryUpdater
from .utils import get_clones, pos_to_pos_embed


class MeMOTR(nn.Module):
    def __init__(
        self,
        backbone: BackboneWithPE,
        transformer: DeformableTransformer,
        query_updater: QueryUpdater,
        num_classes: int,
        n_det_queries: int,
        n_feature_levels: int,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float,
        aux_loss: bool = True,
        with_box_refine: bool = True,
        use_checkpoint: bool = False,
        checkpoint_level: int = 2,
        use_dab: bool = False,
        rot_out_dim=4,
        t_out_dim=3,
        dropout_heads=0.0,
        WITH_BOX_REFINE=True,
        visualize: bool = False,
        use_kpts=False,
        use_kpts_as_ref_pt=False,
        use_kpts_as_img=False,
        use_boxes=True,
        head_num_layers=2,
        head_hidden_dim=256,
        r_num_layers_inc=0,
        use_roi=False,
    ):
        super(MeMOTR, self).__init__()

        self.num_classes = num_classes
        self.n_det_queries = n_det_queries
        self.n_feature_levels = n_feature_levels
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize
        self.head_num_layers = head_num_layers
        self.head_hidden_dim = head_hidden_dim
        self.rot_out_dim = rot_out_dim
        self.t_out_dim = t_out_dim
        self.dropout_heads = dropout_heads
        self.WITH_BOX_REFINE = WITH_BOX_REFINE
        self.use_kpts = use_kpts
        self.use_kpts_as_ref_pt = use_kpts_as_ref_pt
        self.use_kpts_as_img = use_kpts_as_img
        self.use_boxes = use_boxes
        self.use_roi = use_roi

        self.do_predict_2d_t = t_out_dim == 2

        # Net:
        self.backbone = backbone
        self.transformer = transformer
        self.query_updater = query_updater
        self.class_embed = nn.Linear(
            in_features=self.hidden_dim, out_features=num_classes
        )
        if use_boxes:
            self.bbox_embed = MLP(
                in_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                out_dim=4,
                num_layers=3,
            )
        else:
            self.bbox_embed = MLP(
                in_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                out_dim=4,
                num_layers=1,
            )
            for p in self.bbox_embed.parameters():
                p.requires_grad = False

        self.rot_embed = MLP(
            self.hidden_dim * (2 if use_roi else 1),
            hidden_dim=head_hidden_dim,
            out_dim=rot_out_dim,
            num_layers=head_num_layers+r_num_layers_inc,
            dropout=dropout_heads,
            act="relu",
        )
        self.t_embed = MLP(
            self.hidden_dim,
            hidden_dim=head_hidden_dim,
            out_dim=t_out_dim,
            num_layers=head_num_layers,
            dropout=dropout_heads,
            act="relu",
        )
        if self.do_predict_2d_t:
            self.depth_embed = MLP(
                self.hidden_dim,
                hidden_dim=head_hidden_dim,
                out_dim=1,
                num_layers=head_num_layers,
                dropout=dropout_heads,
                act="relu",
            )
        
        if self.use_roi:
            self.roi_cnn = CNNFeatureExtractor(out_dim=self.hidden_dim, model_name="resnet50")

        if self.use_dab:
            self.det_anchor = nn.Parameter(
                torch.randn(self.n_det_queries, 4)
            )  # (N_det, 4)
            self.det_query_embed = nn.Parameter(
                torch.randn(self.n_det_queries, self.hidden_dim)
            )  # (N_det, C)
        else:
            self.det_query_embed = nn.Parameter(
                torch.randn(self.n_det_queries, self.hidden_dim * 2)
            )  # (N_det, 2C)
        assert self.n_feature_levels > 1
        n_backbone_inter_layers = backbone.n_inter_layers()
        n_backbone_inter_channels = backbone.n_inter_channels()
        feature_proj_list = []
        for i in range(n_backbone_inter_layers):
            feature_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=n_backbone_inter_channels[i],
                        out_channels=self.hidden_dim,
                        kernel_size=1,
                    ),
                    nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim),
                )
            )
        for _ in range(self.n_feature_levels - n_backbone_inter_layers):
            feature_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=n_backbone_inter_channels[-1],
                        out_channels=self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim),
                )
            )
        self.feature_projs = nn.ModuleList(feature_proj_list)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.feature_projs:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if self.with_box_refine:
            self.class_embed = get_clones(
                self.class_embed, self.transformer.get_n_dec_layers()
            )
            self.bbox_embed = get_clones(
                self.bbox_embed, self.transformer.get_n_dec_layers()
            )
            self.rot_embed = get_clones(
                self.rot_embed, self.transformer.get_n_dec_layers()
            )
            self.t_embed = get_clones(self.t_embed, self.transformer.get_n_dec_layers())
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.set_refine_bbox_embed(self.bbox_embed)
            if self.do_predict_2d_t:
                self.depth_embed = get_clones(
                    self.depth_embed, self.transformer.get_n_dec_layers()
                )
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(self.transformer.get_n_dec_layers())]
            )
            self.bbox_embed = nn.ModuleList(
                [self.bbox_embed for _ in range(self.transformer.get_n_dec_layers())]
            )
            self.rot_embed = nn.ModuleList(
                [self.rot_embed for _ in range(self.transformer.get_n_dec_layers())]
            )
            self.t_embed = nn.ModuleList(
                [self.t_embed for _ in range(self.transformer.get_n_dec_layers())]
            )
            if self.do_predict_2d_t:
                self.depth_embed = nn.ModuleList(
                    [
                        self.depth_embed
                        for _ in range(self.transformer.get_n_dec_layers())
                    ]
                )

        if self.do_predict_2d_t:
            for p in self.t_embed.parameters():
                p.requires_grad = False

    def forward(self, frame: NestedTensor, tracks: list[TrackInstances]):
        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/memotr/", exist_ok=True)

        # 图像经过 backbone
        if self.use_checkpoint and self.checkpoint_level != 3:
            features, pos = checkpoint(self.backbone, frame, use_reentrant=False)
        else:
            features, pos = self.backbone(frame)

        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.feature_projs[layer](src))
            masks.append(mask)
        if self.n_feature_levels > len(srcs):
            srcs_len = len(srcs)
            for layer in range(srcs_len, self.n_feature_levels):
                if layer == srcs_len:
                    src = self.feature_projs[layer](features[-1].tensors)
                else:
                    src = self.feature_projs[layer](srcs[-1])
                mask = frame.masks
                mask = F.interpolate(mask[None, ...].float(), size=src.shape[-2:])[
                    0
                ].to(torch.bool)
                pos.append(
                    self.backbone.position_embedding(NestedTensor(src, mask)).to(
                        src.device
                    )
                )
                srcs.append(src)
                masks.append(mask)
        # srcs is n_feature_levels * [(B, C, H, W)]
        # masks is n_feature_levels * [(B, H, W)]
        # pos is n_features_levels * [(B, C, H, W)]

        reference_points = self.get_reference_points(tracks=tracks).to(
            srcs[0].device
        )  # (B, Nd+Nq, 2/4)
        query_embed = self.get_query_embed(tracks=tracks).to(srcs[0].device)
        query_mask = self.get_query_mask(tracks=tracks).to(srcs[0].device)  # (B, Nd+Nq)

        # DETR:
        outputs, init_reference, inter_references, inter_queries = self.transformer(
            srcs=srcs,
            masks=masks,
            pos_embeds=pos,
            query_embed=query_embed,
            ref_pts=reference_points,
            query_mask=query_mask,
        )
        # outputs: (n_dec_layers, B, Nd+Nq, C)
        # init_reference: (B, Nd+Nq, 2)
        # inter_references: (n_dec_layers, B, Nd+Nq, 4)
        output_classes, output_bboxes = [], []
        outputs_rots = []
        outputs_ts = []
        outputs_depth = [] if self.do_predict_2d_t else None
        assert outputs.ndim == 4, (
            f"Deformable Transformer's outputs should have shape (n_dec_layers, B, Nd+Nq, C, "
            f"but get n_dim={outputs.ndim}"
        )
        for level in range(outputs.shape[0]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[level - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level](outputs[level])
            bbox_tmp = self.bbox_embed[level](outputs[level])
            if reference.shape[-1] == 4:
                bbox_tmp += reference
            else:
                assert (
                    reference.shape[-1] == 2
                ), f"Reference should have only 2 coord, but get {reference.shape[-1]}."
                bbox_tmp[..., :2] += reference
            output_bbox = bbox_tmp.sigmoid()
            output_classes.append(output_class)
            output_bboxes.append(output_bbox)

            outputs_rot = self.rot_embed[level](outputs[level])
            if self.do_predict_2d_t:
                output_depth = self.depth_embed[level](outputs[level])
                outputs_depth.append(output_depth)
                outputs_t = torch.cat([output_bbox[..., :2], output_depth], dim=-1)
            else:
                outputs_t = self.t_embed[level](outputs[level])
            outputs_rots.append(outputs_rot)
            outputs_ts.append(outputs_t)

            if self.visualize:
                torch.save(
                    reference[0, : self.n_det_queries, :].cpu(),
                    f"./outputs/visualize_tmp/memotr/detection_ref_pts_layer_{level}.tensor",
                )
                torch.save(
                    reference[0, self.n_det_queries :, :].cpu(),
                    f"./outputs/visualize_tmp/memotr/track_ref_pts_layer_{level}.tensor",
                )
                torch.save(
                    output_class[0, : self.n_det_queries, :].cpu(),
                    f"./outputs/visualize_tmp/memotr/detection_logits_layer_{level}.tensor",
                )
                torch.save(
                    output_class[0, self.n_det_queries :, :].cpu(),
                    f"./outputs/visualize_tmp/memotr/track_logits_layer_{level}.tensor",
                )
                torch.save(
                    output_bbox[0, : self.n_det_queries, :].cpu(),
                    f"./outputs/visualize_tmp/memotr/detection_boxes_layer_{level}.tensor",
                )
                torch.save(
                    output_bbox[0, self.n_det_queries :, :].cpu(),
                    f"./outputs/visualize_tmp/memotr/track_boxes_layer_{level}.tensor",
                )

        output_classes = torch.stack(output_classes, dim=0)
        output_bboxes = torch.stack(output_bboxes, dim=0)

        outputs_rot = torch.stack(outputs_rots)
        outputs_t = torch.stack(outputs_ts)

        res = {
            "pred_logits": output_classes[-1],
            "pred_bboxes": output_bboxes[-1],
            "last_ref_pts": (
                inverse_sigmoid(inter_references[-2, :, :, :])
                if self.use_dab  # (B, Nd+Nq, 4)
                else inverse_sigmoid(inter_references[-2, :, :, :])
            ),  # (B, Nd+Nq, 2)
            "query_mask": query_mask,  # (B, Nd+Nq)
            "det_query_embed": query_embed[0][: self.n_det_queries],
            "init_ref_pts": inverse_sigmoid(init_reference),
        }
        if self.do_predict_2d_t:
            outputs_depth = torch.stack(outputs_depth)
            res["center_depth"] = outputs_depth[-1]
        res["rot"] = outputs_rot[-1]
        res["t"] = outputs_t[-1]

        if self.aux_loss:
            res["aux_outputs"] = self.set_aux_loss(
                output_classes=output_classes,
                output_bboxes=output_bboxes,
                outputs_rot=outputs_rot,
                outputs_t=outputs_t,
                outputs_depth=outputs_depth,
                query_mask=query_mask,
                queries=inter_queries,
            )
        res["outputs"] = outputs[-1]  # (B, Nd+Nq, C)
        return res

    @torch.jit.unused
    def set_aux_loss(
        self,
        output_classes,
        output_bboxes,
        query_mask,
        queries,
        outputs_rot=None,
        outputs_t=None,
        outputs_depth=None,
    ):
        """
        this is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        res = [
            {"pred_logits": a, "pred_bboxes": b, "query_mask": query_mask, "queries": c}
            for a, b, c in zip(output_classes[:-1], output_bboxes[:-1], queries[1:])
        ]

        if outputs_rot is not None:
            res = [
                {**x, "rot": r, "t": t}
                for x, r, t in zip(res, outputs_rot[:-1], outputs_t[:-1])
            ]
        if outputs_depth is not None:
            res = [{**x, "center_depth": d} for x, d in zip(res, outputs_depth[:-1])]

        return res

    def get_det_reference_points(self) -> torch.Tensor:
        """
        Returns: (Nd, 2)
        """
        if self.use_dab:
            return self.det_anchor
        else:
            return self.transformer.reference_points(
                self.det_query_embed[:, : self.hidden_dim]
            )

    def get_track_reference_points(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2/4)
        """
        max_len = max([len(t.ref_pts) for t in tracks])
        if self.use_dab:
            references = torch.zeros((len(tracks), max_len, 4))
        else:
            # references = torch.zeros((len(tracks), max_len, 2))
            references = (
                torch.zeros((len(tracks), max_len, 4))
                if self.WITH_BOX_REFINE
                else torch.zeros((len(tracks), max_len, 2))
            )
        for i in range(len(tracks)):
            references[i, : len(tracks[i].ref_pts), :] = tracks[i].ref_pts
        return references

    def get_track_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2C)
        """
        max_len = max([len(t.query_embed) for t in tracks])
        if self.use_dab:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim))
        else:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim * 2))
        for i in range(len(tracks)):
            query_embed[i, : len(tracks[i].query_embed), :] = tracks[i].query_embed
        return query_embed

    def get_reference_points(self, tracks: list[TrackInstances]):
        det_references = self.get_det_reference_points().repeat(
            len(tracks), 1, 1
        )  # (B, Nd, 2)
        if det_references.shape[-1] == 2:
            det_references = torch.cat(
                (
                    det_references,
                    torch.zeros_like(det_references, device=det_references.device),
                ),
                dim=-1,
            )
        track_references = self.get_track_reference_points(tracks=tracks).to(
            det_references.device
        )  # (B, Nq, 2)
        return torch.cat((det_references, track_references), dim=1)

    def get_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nd+Nq, 2C)
        """
        if self.use_dab:
            det_query_embed = self.det_query_embed
            det_query_embed = det_query_embed.repeat(len(tracks), 1, 1)
        else:
            det_query_embed = self.det_query_embed.repeat(
                len(tracks), 1, 1
            )  # (B, Nd, 2C)
        track_query_embed = self.get_track_query_embed(tracks).to(
            det_query_embed.device
        )  # (B, Nq, 2C)
        return torch.cat((det_query_embed, track_query_embed), dim=1)

    def get_query_mask(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nd+Nq)
        """
        track_max_len = max([len(t.query_embed) for t in tracks])
        det_query_mask = torch.zeros((len(tracks), self.n_det_queries)).to(torch.bool)
        track_query_mask = torch.zeros((len(tracks), track_max_len))
        for i in range(len(tracks)):
            if len(tracks[i].query_embed) > 0:
                track_query_mask[i, len(tracks[i].query_embed) :] = 1
        track_query_mask = track_query_mask.to(torch.bool)
        return torch.cat((det_query_mask, track_query_mask), dim=1).to(
            self.det_query_embed.device
        )

    def postprocess_single_frame(
        self,
        previous_tracks: List[TrackInstances],
        new_tracks: List[TrackInstances],
        unmatched_dets: List[TrackInstances] | None,
        no_augment: bool = False,
    ):
        """
        Query updating.
        """
        return self.query_updater(
            previous_tracks, new_tracks, unmatched_dets, no_augment
        )


def build(config: dict, num_classes=None):
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
    }
    assert (
        config["DATASET"] in dataset_num_classes
    ), f"Do not know the class num of {config['DATASET']} dataset."
    num_classes = num_classes or dataset_num_classes[config["DATASET"]]
    t_out_dim = config.get("t_out_dim", 3)
    dropout = config.get("dropout", 0.0)
    dropout_heads = config.get("dropout_heads", 0.0)
    use_kpts = config.get("use_kpts", False)
    use_kpts_as_ref_pt = config.get("use_kpts_as_ref_pt", False)
    use_kpts_as_img = config.get("use_kpts_as_img", False)
    opt_only = config.get("opt_only", None)
    use_boxes = opt_only is None or (opt_only is not None and all(x in opt_only for x in ["boxes"]))
    head_num_layers = config.get("head_num_layers", 2)
    head_hidden_dim = config.get("head_hidden_dim", 256)

    backbone_with_pe = build_backbone_with_pe(config=config)
    deformable_transformer = build_deformable_transformer(config=config)
    query_updater = build_query_updater(config=config)

    return MeMOTR(
        backbone=backbone_with_pe,
        transformer=deformable_transformer,
        query_updater=query_updater,
        num_classes=num_classes,
        n_det_queries=config["NUM_DET_QUERIES"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        aux_loss=True,
        with_box_refine=config.get("WITH_BOX_REFINE", True),
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"],
        rot_out_dim=config.get("rot_out_dim", 4),
        t_out_dim=t_out_dim,
        dropout=dropout,
        WITH_BOX_REFINE=config.get("WITH_BOX_REFINE", True),
        dropout_heads=dropout_heads,
        use_kpts=use_kpts,
        use_kpts_as_ref_pt=use_kpts_as_ref_pt,
        use_kpts_as_img=use_kpts_as_img,
        use_boxes=use_boxes,
        head_num_layers=head_num_layers,
        head_hidden_dim=head_hidden_dim,
        r_num_layers_inc=config.get("r_num_layers_inc", 0),
        use_roi=config.get("use_roi", False),
    )


def build_model(config: dict, num_classes=None):
    from memotr.utils.utils import distributed_rank
    model = build(config=config, num_classes=num_classes)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model

if __name__ == "__main__":
    ...
