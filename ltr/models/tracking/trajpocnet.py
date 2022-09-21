import math
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.transformer.transformer as trans
import ltr.models.transformer.traj_filter_predictor as tfp
import ltr.models.transformer.heads as heads
from ltr import MultiGPU
from ltr.models.layers.normalization import InstanceL2Norm


class TrajPOCnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, head):
        super().__init__()

        # self.feature_extractor = feature_extractor
        self.head = head
        # self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        # self.output_layers = sorted(list(set(self.head_layer)))

    def forward(self, cur_img, prev_img, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert cur_img.dim() == 5 and prev_img.dim() == 5, 'Expect 5 dimensional inputs'

        # # Extract backbone features
        # train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        # test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))
        #
        # # Classification features
        # train_feat_head = self.get_backbone_head_feat(train_feat)
        # test_feat_head = self.get_backbone_head_feat(test_feat)

        # Run head module
        test_scores = self.head(train_feat=prev_img, test_feat=cur_img, train_bb=None, no_bbox=True, *args, **kwargs)

        return test_scores

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def trajpocnet(out_feature_dim=512, head_feat_blocks=0, head_feat_norm=True, final_conv=True, nhead=8,
               num_encoder_layers=6,
               num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # filter_size = 4, head_layer = 'layer3', backbone_pretrained = True, head_feat_blocks = 0, head_feat_norm = True,
    # final_conv = True, out_feature_dim = 512, frozen_backbone_layers = (), nhead = 8, num_encoder_layers = 6,
    # num_decoder_layers = 6, dim_feedforward = 2048, feature_sz = 18, use_test_frame_encoding = True, multi_gpu_head = False
    # Backbone
    # backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim))

    # Classifier features
    # if head_layer == 'layer3':
    # feature_dim = 3
    # elif head_layer == 'layer4':
    #     feature_dim = 512
    # else:
    #     raise Exception

    # head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
    #                                                           num_blocks=head_feat_blocks, l2norm=head_feat_norm,
    #                                                           final_conv=final_conv, norm_scale=norm_scale,
    #                                                           out_dim=out_feature_dim)

    head_feature_extractor = nn.Sequential(
        nn.Conv2d(3, out_feature_dim, kernel_size=1, padding=0, bias=True),
        InstanceL2Norm(scale=norm_scale))

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = tfp.FilterPredictor(transformer, feature_sz=feature_sz,
                                           use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    # bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=None)

    # ToMP network
    net = TrajPOCnet(head=head)
    return net
