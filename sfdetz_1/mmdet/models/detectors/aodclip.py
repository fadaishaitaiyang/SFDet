# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.models import builder
from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from typing import Tuple, Union
import os
import matplotlib.pyplot as plt


from typing import Any, Union, List
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
@DETECTORS.register_module()
class AodClip(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 text_encoder,
                 pretrained_text,
                 
                 context_length,
                 base_class,
                 novel_class,
                 both_class,
                 tau=0.07,
                 multi_prompts=False,
                 self_training=False,
                 ft_backbone=False,
                 exclude_key=None,
                 load_text_embedding=None,
                #  init_cfg=None,
                 **args):
        super(AodClip, self).__init__(
            **args)
        if pretrained_text is not None:
            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = pretrained_text

        self.text_encoder = builder.build_backbone(text_encoder)

        self.tau = tau
        self.class_names = ('echinus','starfish','holothurian','scallop')

        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)
        self.self_training = self_training
        self.multi_prompts = multi_prompts
        self.load_text_embedding = load_text_embedding

        if not self.load_text_embedding:
            if not self.multi_prompts:
                self.texts = torch.cat([tokenize(f"a photo of a {c}") for c in self.class_names]) 
            else:
                self.texts = self._get_multi_prompts(self.class_names)

        if len(self.base_class) != len(self.both_class): # zero-shot setting
            if not self_training:
                self._visiable_mask(self.base_class, self.novel_class, self.both_class)
            else:
                self._visiable_mask_st(self.base_class, self.novel_class, self.both_class)
                self._st_mask(self.base_class, self.novel_class, self.both_class)

        if self.training:
            self._freeze_stages(self.text_encoder)
            if ft_backbone is False:
                self._freeze_stages(self.backbone, exclude_key=exclude_key)

        else:
            self.text_encoder.eval()
            self.backbone.eval()
            
    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count>0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def _visiable_mask(self, seen_classes, novel_classes, both_classes):
        seen_map = np.array([-1]*256)
        seen_map[255] = 255
        for i,n in enumerate(list(seen_classes)):
            seen_map[n] = i
        self.visibility_seen_mask = seen_map.copy()
        print('Making visible mask for zero-shot setting:', self.visibility_seen_mask) 
    
    def _visiable_mask_st(self, seen_classes, novel_classes, both_classes):
        seen_map = np.array([-1]*256)
        seen_map[255] = 255
        for i,n in enumerate(list(seen_classes)):
            seen_map[n] = n
        seen_map[200] = 200 # pixels of padding will be excluded
        self.visibility_seen_mask = seen_map.copy()
        print('Making visible mask for zero-shot setting in self_traning stage:', self.visibility_seen_mask) 
    
    def _st_mask(self, seen_classes, novel_classes, both_classes):
        st_mask  = np.array([255]*256)
        st_mask[255] = 255
        for i,n in enumerate(list(novel_classes)):
            st_mask[n] = n
        self.st_mask = st_mask.copy()
        print('Making st mask for zero-shot setting in self_traning stage:', self.st_mask) 
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        visual = self.backbone(img)
        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding)
            text_feat = torch.from_numpy(text_feat).to(img.device)
        else:
            if not self.multi_prompts:
                text_feat = self.text_embedding(self.texts, img)
            else:
                assert AttributeError("preparing the multi embeddings")

        if not self.self_training:
            text_feat = text_feat[self.base_class, :]
        x =[]
        x.append(visual)
        x.append(text_feat)
        if self.with_neck:
            x = self.neck(x)
        return x
    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs
    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN, self).show_result(data, result, **kwargs)
