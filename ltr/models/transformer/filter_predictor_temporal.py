import time

import torch
import torch.nn as nn
from ltr.models.transformer.position_encoding import PositionEmbeddingSine, PositionEmbeddingSine3D


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FilterPredictorTemporal(nn.Module):
    def __init__(self, transformer, feature_sz, num_max_frames=20, gap=5, num_train_ims=2,use_test_frame_encoding=True):
        super().__init__()
        self.num_max_frames = num_max_frames
        self.gap = gap
        self.num_train_ims = num_train_ims
        self.transformer = transformer
        self.feature_sz = feature_sz
        self.use_test_frame_encoding = use_test_frame_encoding

        self.box_encoding = MLP([4, self.transformer.d_model//4, self.transformer.d_model, self.transformer.d_model])

        self.query_embed_fg = nn.Embedding(1, self.transformer.d_model)

        if self.use_test_frame_encoding:
            self.query_embed_test = nn.Embedding(1, self.transformer.d_model)

        self.query_embed_fg_decoder = self.query_embed_fg

        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=self.transformer.d_model//2, sine_type='lin_sine',
                                                  avoid_aliazing=True, max_spatial_resolution=feature_sz)
        self.pos_encoding3D = PositionEmbeddingSine3D(num_pos_feats=self.transformer.d_model // 2, sine_type='lin_sine',
                                                  avoid_aliazing=True, max_spatial_resolution=torch.tensor([feature_sz, feature_sz, self.num_train_ims*self.gap + self.num_max_frames + 1]))
        self.pos_embs = None

    def forward(self, train_feat, test_feat, train_label, train_ltrb_target, *args, **kwargs):
        return self.predict_filter(train_feat, test_feat, train_label, train_ltrb_target, *args, **kwargs)

    def get_positional_encoding(self, feat):
        nframes, nseq, _, h, w = feat.shape

        mask = torch.zeros((nframes * nseq, h, w), dtype=torch.bool, device=feat.device)
        pos = self.pos_encoding(mask)

        return pos.reshape(nframes, nseq, -1, h, w)

    def get_positional_encoding3D(self, nframes, nseq, h, w, device):
        # nframes, nseq, _, h, w = feat.shape

        mask = torch.zeros((nframes, nseq, h, w), dtype=torch.bool, device=device)
        pos = self.pos_encoding3D(mask)

        return pos

    def predict_filter(self, train_feat, test_feat, train_label, train_ltrb_target, temporal_features, *args, **kwargs):
        def printv(x, y):
            if y.dtype != torch.bool:
                print(
                    f"{x}: {y.shape} , mean:{y.mean():.4f}, min:{y.min():.4f}, max:{y.max():.4f}, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")
            else:
                print(
                    f"{x}: {y.shape} , mean:-----, min:-----, max:-----, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")
        debug = False

        #train_label size guess: Nf_tr, Ns, H, W.
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)

        h, w = test_feat.shape[-2:]
        B = test_feat.shape[1]

        # test_pos = self.get_positional_encoding(test_feat) # Nf_te, Ns, C, H, W
        # train_pos = self.get_positional_encoding(train_feat) # Nf_tr, Ns, C, H, W

        if self.pos_embs is None:
            #  frames are from among first gap*num_train_ims frames are for selected train images (they are in pos=0, pos=gap) [0, gap, gap*2,...]
            #  from [n-1-self.num_max_frames-1, -2] are for temporal features
            # [-1] is for test image
            # shape = n_ims x B x h x w x dim
            self.pos_embs = self.get_positional_encoding3D(self.num_train_ims*self.gap + self.num_max_frames + 1, test_feat.shape[1], h, w, train_feat.device)
        if debug:
            printv("self.pos_embs", self.pos_embs)
            print("\n\n")
            printv("FP:train_feat", train_feat)
            printv("FP:test_feat", test_feat)
            printv("FP:train_label", train_label)
            printv("FP:train_ltrb_target", train_ltrb_target)
            if temporal_features is not None:
                printv('features', temporal_features['feats'])
                printv('labels', temporal_features['labels'])
                printv('ltrb', temporal_features['ltrb'])
                print(temporal_features['mask'].shape)
                print(temporal_features['labels'].requires_grad, train_label.requires_grad)

        # i x b x c x h x w  => b x c x i x h x w
        test_feat_seq = test_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1) # Nf_te*H*W, Ns, C
        train_feat_seq = train_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1) # Nf_tr*H*W, Ns, C
        train_label_seq = train_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2) # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2) # Ns,4,Nf_tr*H*W

        # test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        # train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)

        # printv('self.pos_embs', self.pos_embs)
        # 0 1 2 3 4 => 1 4 0 2 3
        # (txbxhxwxdim) =permute(1 4 0 2 3)=> (bxdimxtxhxw)
        # (bxdimxhxw) =flatten(2)=> (bxdimx(h*w))
        num_tests = test_feat.shape[0]
        test_pos = self.pos_embs[[-1]*num_tests,:B, ...].permute(1, 4, 0, 2, 3).flatten(2).permute(2,0,1)

        # (n_imsxbxhxwxdim) =permute(1,4,0,2,3)=> (bxdimxhxw)
        # (bxdimxhxw) =flatten(2)=> (bxdimx(h*w))
        train_pos = self.pos_embs[[i*self.gap for i in range(self.num_train_ims)],:B, ...].permute(1,4,0,2,3).flatten(2).permute(2,0,1)


        # foreground embedding
        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq # Nf_tr*H*W,Ns,C


        if temporal_features is not None:
            # stime = time.time()
            t_feats = temporal_features['feats'] # txbxdimxhxw
            t_label = temporal_features['labels'] # txbxhxw
            t_ltrb = temporal_features['ltrb']  # txbx4xhxw
            t_mask = temporal_features['mask'] # txbxhxw
            b = t_feats.shape[1]
            t_frames = t_feats.shape[0]
            temporal_feat_seq = torch.stack([t_feats[:,b_i:b_i+1,...].permute(0,1,3,4,2)[t_mask[:,b_i:b_i+1,...]] for b_i in range(b)], dim=1) # num_feat x b x C
            temporal_feat_seq = temporal_feat_seq.detach()
            temporal_pos = torch.stack([self.pos_embs[-1-t_frames:-1,b_i:b_i+1,...][t_mask[:,b_i:b_i+1,...]] for b_i in range(b)], dim=1)
            temporal_label_enc = fg_token * torch.stack([t_label[:,b_i:b_i+1,...][t_mask[:,b_i:b_i+1,...]] for b_i in range(b)], dim=1).unsqueeze(2) # num_feat x b x 1
            temp_ltrb_from_mask = torch.stack([t_ltrb[:,b_i:b_i+1,...].permute(0,1,3,4,2)[t_mask[:,b_i:b_i+1,...]] for b_i in range(b)], dim=1) # num_feat x b x 4
            temporal_ltrb_enc = self.box_encoding(temp_ltrb_from_mask.permute(1,2,0)).permute(2,0,1) # num_feat x b x C
            # if debug:
            # print(f"Temporal feat makeup {time.time() - stime}")




        # printv('train_ltrb_target_seq_T', train_ltrb_target_seq_T)
        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2,0,1) # Nf_tr*H*H,Ns,C
        #
        # printv('train_feat_seq', train_feat_seq)
        # printv('train_label_enc', train_label_enc)
        # printv('train_ltrb_target_enc', train_ltrb_target_enc)
        # printv('test_feat_seq', test_feat_seq)

        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token


        if debug:
            printv('train_feat_seq', train_feat_seq)
            printv('train_label_enc', train_label_enc)
            printv('train_ltrb_target_enc', train_ltrb_target_enc)
            printv('train_pos', train_pos)
            if (temporal_features is not None):
                printv('temporal_feat_seq', temporal_feat_seq)
                printv('temporal_label_enc', temporal_label_enc)
                printv('temporal_ltrb_enc', temporal_ltrb_enc)
                printv('temporal_pos', temporal_pos)
            printv('test_feat_seq', test_feat_seq)
            printv('test_pos', test_pos)

        if temporal_features is not None:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, temporal_feat_seq + temporal_label_enc + temporal_ltrb_enc ,  (test_feat_seq + test_label_enc) if self.use_test_frame_encoding else test_feat_seq], dim=0)
            pos = torch.cat([train_pos, temporal_pos,test_pos], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, (test_feat_seq + test_label_enc) if self.use_test_frame_encoding else test_feat_seq], dim=0)
            pos = torch.cat([train_pos, test_pos], dim=0)

        if debug:
            printv('feat', feat)
            printv('pos', pos)
        # print("Transformer Input shapes", feat.shape, pos.shape)
        output_embed, enc_mem = self.transformer(feat, mask=None, query_embed=self.query_embed_fg_decoder.weight, pos_embed=pos)

        enc_opt = enc_mem[-h*w:].transpose(0, 1)
        dec_opt = output_embed.squeeze(0).transpose(1, 2)

        return dec_opt.reshape(test_feat.shape[1], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(test_feat.shape)

    def predict_cls_bbreg_filters_parallel_bkp(self, train_feat, test_feat, train_label, num_gth_frames, train_ltrb_target, *args, **kwargs):
        # train_label size guess: Nf_tr, Ns, H, W.
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)

        h, w = test_feat.shape[-2:]
        H, W = train_feat.shape[-2:]

        train_feat_stack = torch.cat([train_feat, train_feat], dim=1)
        test_feat_stack = torch.cat([test_feat, test_feat], dim=1)
        train_label_stack = torch.cat([train_label, train_label], dim=1)
        train_ltrb_target_stack = torch.cat([train_ltrb_target, train_ltrb_target], dim=1)

        test_pos = self.get_positional_encoding(test_feat)  # Nf_te, Ns, C, H, W
        train_pos = self.get_positional_encoding(train_feat)  # Nf_tr, Ns, C, H, W

        test_feat_seq = test_feat_stack.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_te*H*W, Ns, C
        train_feat_seq = train_feat_stack.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_tr*H*W, Ns, C
        train_label_seq = train_label_stack.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2)  # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target_stack.permute(1, 2, 0, 3, 4).flatten(2)  # Ns,4,Nf_tr*H*W

        test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)

        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq

        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2, 0, 1)  # Nf_tr*H*H,Ns,C

        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq], dim=0)

        pos = torch.cat([train_pos, test_pos], dim=0)

        src_key_padding_mask = torch.zeros(feat.shape[1], feat.shape[0]).bool()
        src_key_padding_mask[1, num_gth_frames*H*W:-h*w] = 1.
        src_key_padding_mask = src_key_padding_mask.bool().to(feat.device)

        output_embed, enc_mem = self.transformer(feat, mask=src_key_padding_mask,
                                                 query_embed=self.query_embed_fg_decoder.weight,
                                                 pos_embed=pos)

        enc_opt = enc_mem[-h * w:].transpose(0, 1).permute(0, 2, 1).reshape(test_feat_stack.shape)
        dec_opt = output_embed.squeeze(0).transpose(1, 2).reshape(test_feat_stack.shape[1], -1, 1, 1)

        cls_enc_opt = enc_opt[:, 0].unsqueeze(1)
        bbreg_enc_opt = enc_opt[:, 1].unsqueeze(1)
        cls_dec_opt = dec_opt[0].unsqueeze(0)
        bbreg_dec_opt = dec_opt[1].unsqueeze(0)

        return cls_dec_opt, bbreg_dec_opt, cls_enc_opt, bbreg_enc_opt

    # _simple_impl
    def predict_cls_bbreg_filters_parallel_simple_impl(self, train_feat, test_feat, train_label, num_gth_frames, train_ltrb_target, *args, **kwargs):
        # train_label size guess: Nf_tr, Ns, H, W.
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)

        h, w = test_feat.shape[-2:]
        H, W = train_feat.shape[-2:]


        test_pos = self.get_positional_encoding(test_feat)  # Nf_te, Ns, C, H, W
        train_pos = self.get_positional_encoding(train_feat)  # Nf_tr, Ns, C, H, W

        test_feat_seq = test_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_te*H*W, Ns, C
        train_feat_seq = train_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_tr*H*W, Ns, C
        train_label_seq = train_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2)  # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2)  # Ns,4,Nf_tr*H*W

        test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)

        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq

        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2, 0, 1)  # Nf_tr*H*H,Ns,C

        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq], dim=0)

        pos = torch.cat([train_pos, test_pos], dim=0)

        feat = torch.cat([feat]*2, dim=1)
        src_key_padding_mask = torch.zeros(feat.shape[1], feat.shape[0]).bool()
        src_key_padding_mask[1, num_gth_frames*H*W:-h*w] = 1.
        src_key_padding_mask = src_key_padding_mask.bool().to(feat.device)

        output_embed, enc_mem = self.transformer(feat, mask=src_key_padding_mask,
                                                 query_embed=self.query_embed_fg_decoder.weight,
                                                 pos_embed=pos)
        _, _, e, h, w = test_feat.shape
        enc_opt = enc_mem[-h * w:].transpose(0, 1).permute(0, 2, 1).reshape((1,2,e,h,w))
        dec_opt = output_embed.squeeze(0).transpose(1, 2).reshape(2, -1, 1, 1)

        cls_enc_opt = enc_opt[:, 0].unsqueeze(1)
        bbreg_enc_opt = enc_opt[:, 1].unsqueeze(1)
        cls_dec_opt = dec_opt[0].unsqueeze(0)
        bbreg_dec_opt = dec_opt[1].unsqueeze(0)
        return cls_dec_opt, bbreg_dec_opt, cls_enc_opt, bbreg_enc_opt

    def predict_cls_bbreg_filters_parallel(self, train_feat, test_feat, train_label, num_gth_frames, train_ltrb_target, temporal_features=None, misplaced_train_frame_pos=False, *args, **kwargs):
        def printv(x, y):
            if y.dtype != torch.bool:
                print(
                    f"{x}: {y.shape} , mean:{y.mean():.4f}, min:{y.min():.4f}, max:{y.max():.4f}, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")
            else:
                print(
                    f"{x}: {y.shape} , mean:-----, min:-----, max:-----, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")
        debug = False

        #train_label size guess: Nf_tr, Ns, H, W.
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)

        h, w = test_feat.shape[-2:]
        B = test_feat.shape[1]
        H, W = train_feat.shape[-2:]

        # test_pos = self.get_positional_encoding(test_feat) # Nf_te, Ns, C, H, W
        # train_pos = self.get_positional_encoding(train_feat) # Nf_tr, Ns, C, H, W

        if self.pos_embs is None:
            #  frames are from among first gap*num_train_ims frames are for selected train images (they are in pos=0, pos=gap) [0, gap, gap*2,...]
            #  from [n-1-self.num_max_frames-1, -2] are for temporal features
            # [-1] is for test image
            # shape = n_ims x B x h x w x dim
            self.pos_embs = self.get_positional_encoding3D(self.num_train_ims*self.gap + self.num_max_frames + 1, test_feat.shape[1], h, w, train_feat.device)
        if debug:
            printv("self.pos_embs", self.pos_embs)
            print("\n\n")
            printv("FP:train_feat", train_feat)
            printv("FP:test_feat", test_feat)
            printv("FP:train_label", train_label)
            printv("FP:train_ltrb_target", train_ltrb_target)
            if temporal_features is not None:
                printv('features', temporal_features['feats'])
                printv('labels', temporal_features['labels'])
                printv('ltrb', temporal_features['ltrb'])
                print(temporal_features['mask'].shape)
                print(temporal_features['labels'].requires_grad, train_label.requires_grad)

        # i x b x c x h x w  => b x c x i x h x w
        test_feat_seq = test_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1) # Nf_te*H*W, Ns, C
        train_feat_seq = train_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1) # Nf_tr*H*W, Ns, C
        train_label_seq = train_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2) # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2) # Ns,4,Nf_tr*H*W

        # test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        # train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)

        # printv('self.pos_embs', self.pos_embs)
        # 0 1 2 3 4 => 1 4 0 2 3
        # (txbxhxwxdim) =permute(1 4 0 2 3)=> (bxdimxtxhxw)
        # (bxdimxhxw) =flatten(2)=> (bxdimx(h*w))
        num_tests = test_feat.shape[0]
        test_pos = self.pos_embs[[-1]*num_tests,:B, ...].permute(1, 4, 0, 2, 3).flatten(2).permute(2,0,1)

        # (n_imsxbxhxwxdim) =permute(1,4,0,2,3)=> (bxdimxhxw)
        # (bxdimxhxw) =flatten(2)=> (bxdimx(h*w))
        offset = 2 if misplaced_train_frame_pos else 0
        train_pos = self.pos_embs[[i*self.gap+offset for i in range(train_feat.shape[0])],:B, ...].permute(1,4,0,2,3).flatten(2).permute(2,0,1)


        # foreground embedding
        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq # Nf_tr*H*W,Ns,C


        if temporal_features is not None:
            # stime = time.time()
            t_feats = temporal_features['feats'] # txbxdimxhxw
            t_label = temporal_features['labels'] # txbxhxw
            t_ltrb = temporal_features['ltrb']  # txbx4xhxw
            t_mask = temporal_features['mask'] # txbxhxw
            if t_mask.sum() == 0:
                temporal_features = None
            else:
                b = t_feats.shape[1]
                t_frames = t_feats.shape[0]
                temporal_feat_seq = torch.stack([t_feats[:,b_i:b_i+1,...].permute(0,1,3,4,2)[t_mask[:,b_i:b_i+1,...]] for b_i in range(b)], dim=1) # num_feat x b x C
                temporal_feat_seq = temporal_feat_seq.detach()
                temporal_pos = torch.stack([self.pos_embs[-1-t_frames:-1,b_i:b_i+1,...][t_mask[:,b_i:b_i+1,...]] for b_i in range(b)], dim=1)
                temporal_label_enc = fg_token * torch.stack([t_label[:,b_i:b_i+1,...][t_mask[:,b_i:b_i+1,...]] for b_i in range(b)], dim=1).unsqueeze(2) # num_feat x b x 1
                temp_ltrb_from_mask = torch.stack([t_ltrb[:,b_i:b_i+1,...].permute(0,1,3,4,2)[t_mask[:,b_i:b_i+1,...]] for b_i in range(b)], dim=1) # num_feat x b x 4
                temporal_ltrb_enc = self.box_encoding(temp_ltrb_from_mask.permute(1,2,0)).permute(2,0,1) # num_feat x b x C
            # if debug:
            # print(f"Temporal feat makeup {time.time() - stime}")




        # printv('train_ltrb_target_seq_T', train_ltrb_target_seq_T)
        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2,0,1) # Nf_tr*H*H,Ns,C
        #
        # printv('train_feat_seq', train_feat_seq)
        # printv('train_label_enc', train_label_enc)
        # printv('train_ltrb_target_enc', train_ltrb_target_enc)
        # printv('test_feat_seq', test_feat_seq)

        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token


        if debug:
            printv('train_feat_seq', train_feat_seq)
            printv('train_label_enc', train_label_enc)
            printv('train_ltrb_target_enc', train_ltrb_target_enc)
            printv('train_pos', train_pos)
            if (temporal_features is not None):
                printv('temporal_feat_seq', temporal_feat_seq)
                printv('temporal_label_enc', temporal_label_enc)
                printv('temporal_ltrb_enc', temporal_ltrb_enc)
                printv('temporal_pos', temporal_pos)
            printv('test_feat_seq', test_feat_seq)
            printv('test_pos', test_pos)

        if temporal_features is not None:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, temporal_feat_seq + temporal_label_enc + temporal_ltrb_enc ,  (test_feat_seq + test_label_enc) if self.use_test_frame_encoding else test_feat_seq], dim=0)
            pos = torch.cat([train_pos, temporal_pos,test_pos], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, (test_feat_seq + test_label_enc) if self.use_test_frame_encoding else test_feat_seq], dim=0)
            pos = torch.cat([train_pos, test_pos], dim=0)

        feat = torch.cat([feat]*2, dim=1)
        src_key_padding_mask = torch.zeros(feat.shape[1], feat.shape[0]).bool()
        src_key_padding_mask[1, num_gth_frames * H * W:-h * w] = 1.
        src_key_padding_mask = src_key_padding_mask.bool().to(feat.device)

        if debug:
            printv('feat', feat)
            printv('pos', pos)
        # print("Transformer Input shapes", feat.shape, pos.shape)
        output_embed, enc_mem = self.transformer(feat, mask=src_key_padding_mask, query_embed=self.query_embed_fg_decoder.weight, pos_embed=pos)
        _, _, e, h, w = test_feat.shape
        enc_opt = enc_mem[-h * w:].transpose(0, 1).permute(0, 2, 1).reshape((1,2,e,h,w))
        dec_opt = output_embed.squeeze(0).transpose(1, 2).reshape(2, -1, 1, 1)

        cls_enc_opt = enc_opt[:, 0].unsqueeze(1)
        bbreg_enc_opt = enc_opt[:, 1].unsqueeze(1)
        cls_dec_opt = dec_opt[0].unsqueeze(0)
        bbreg_dec_opt = dec_opt[1].unsqueeze(0)
        return cls_dec_opt, bbreg_dec_opt, cls_enc_opt, bbreg_enc_opt


        # enc_opt = enc_mem[-h*w:].transpose(0, 1)
        # dec_opt = output_embed.squeeze(0).transpose(1, 2)
        #
        # return dec_opt.reshape(test_feat.shape[1], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(test_feat.shape)

