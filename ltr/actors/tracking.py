from pytracking import TensorDict
from . import BaseActor
import torch
import os
from tqdm import tqdm

from ..admin.multigpu import is_multi_gpu

iou_char = ['˅', ' ', '_', '░', '▒', '▓', '█']
iou_range_split = [-999999.0, 0, .02, .05, 0.15, .25, .4]


class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in
                               target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2, -1)) for s in
                             target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
               loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats


class KYSActor(BaseActor):
    """ Actor for training KYS model """

    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        self.dimp_jitter_fn = dimp_jitter_fn

        # TODO set it somewhere
        self.device = torch.device("cuda:0")

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Initialize loss variables
        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_orig_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        # Initialize target model using the training frames
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # Track in the first test frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all).view(1, num_sequences, -1,
                                                                                           backbone_feat_prev.shape[-2],
                                                                                           backbone_feat_prev.shape[-1])
        else:
            motion_feat_prev = backbone_feat_prev

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col (added due to even kernel size in the target model)
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        # Set previous frame information
        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)
        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        # Loop over the sequence
        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i + 1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            # Extract features
            backbone_feat_cur_all = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.net.motion_feat_extractor is not None:
                motion_feat_cur = self.net.motion_feat_extractor(backbone_feat_cur_all).view(1, num_sequences, -1,
                                                                                             backbone_feat_cur.shape[
                                                                                                 -2],
                                                                                             backbone_feat_cur.shape[
                                                                                                 -1])
            else:
                motion_feat_cur = backbone_feat_cur

            # Run target model
            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            # Jitter target model output for augmentation
            jitter_info = None
            if self.dimp_jitter_fn is not None:
                dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

            # Input target model output along with previous frame information to the predictor
            predictor_input_data = {'input1': motion_feat_prev, 'input2': motion_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (
                    test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * ~uncertain_frame

            # Calculate losses
            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if 'fused_score_orig' in aux_data and 'test_clf_orig' in self.loss_weight.keys():
                aux_data['fused_score_orig'] = aux_data['fused_score_orig'].view(test_label_cur.shape)
                clf_loss_test_orig_new = self.objective['test_clf'](aux_data['fused_score_orig'], test_label_cur,
                                                                    test_anno_cur, valid_samples=is_valid)
                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur,
                                                                            valid_samples=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur,
                                                                            valid_samples=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            motion_feat_prev = motion_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        # Compute average loss over the sequence
        clf_loss_test = clf_loss_test_all.mean()
        clf_loss_test_orig = clf_loss_test_orig_all.mean()
        dimp_loss_test = dimp_loss_test_all.mean()
        is_target_loss = is_target_loss_all.mean()
        is_target_after_prop_loss = is_target_after_prop_loss_all.mean()

        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        clf_loss_test_orig /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        clf_loss_test_orig_w = self.loss_weight['test_clf_orig'] * clf_loss_test_orig
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w + clf_loss_test_orig_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/test_clf_orig': clf_loss_test_orig.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        return loss, stats


class DiMPSimpleActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'],
                                            train_label=data['train_label'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        loss_test_init_clf = 0
        loss_test_iter_clf = 0

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Loss for the initial filter iteration
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class TargetCandiateMatchingActor(BaseActor):
    """Actor for training the KeepTrack network."""

    def __init__(self, net, objective):
        super().__init__(net, objective)

    def __call__(self, data):
        """
        args:
            data - The input data.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

        preds = self.net(**data)

        # Classification losses for the different optimization iterations
        losses = self.objective['target_candidate_matching'](**data, **preds)

        # Total loss
        loss = losses['total'].mean()

        # Log stats
        stats = {
            'Loss/total': loss.item(),
            'Loss/nll_pos': losses['nll_pos'].mean().item(),
            'Loss/nll_neg': losses['nll_neg'].mean().item(),
            'Loss/num_matchable': losses['num_matchable'].mean().item(),
            'Loss/num_unmatchable': losses['num_unmatchable'].mean().item(),
            'Loss/sinkhorn_norm': losses['sinkhorn_norm'].mean().item(),
            'Loss/bin_score': losses['bin_score'].item(),
        }

        if hasattr(self.objective['target_candidate_matching'], 'metrics'):
            metrics = self.objective['target_candidate_matching'].metrics(**data, **preds)

            for key, val in metrics.items():
                stats[key] = torch.mean(val[~torch.isnan(val)]).item()

        return loss, stats


class ToMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bbox_preds = self.net(train_imgs=data['train_images'],
                                             test_imgs=data['test_images'],
                                             train_bb=data['train_anno'],
                                             train_label=data['train_label'],
                                             train_ltrb_target=data['train_ltrb_target'])

        loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious > 0].std().item()

        return loss, stats


class MyToMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        num_frames = data['test_images'].shape[0]
        loss_all = []
        stats = {}
        loss = None
        for i in tqdm(range(num_frames)):
            if loss is not None and i % 25 == 0:
                # print("Running backward")
                loss.backward()
                loss = 0

            if loss is None:
                loss = 0
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size(), obj.get_device())
            #     except:
            #         pass
            # print(torch.cuda.memory_summary())
            # print(f"Frame: {i} is now running")
            # Run network
            target_scores, bbox_preds = self.net(train_imgs=data['train_images'],
                                                 test_imgs=data['test_images'][i][None],
                                                 train_bb=data['train_anno'],
                                                 train_label=data['train_label'],
                                                 train_ltrb_target=data['train_ltrb_target'],
                                                 prev_features=None)

            loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'][i][None],
                                                     data['test_sample_region'][i][None])

            # Classification losses for the different optimization iterations
            clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'][i][None],
                                                       data['test_anno'][i][None])

            loss += self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test
            # loss_all.append(loss)
            if torch.isnan(loss):
                raise ValueError('NaN detected in loss')

            ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'][i][None],
                                                                 bbox_preds)

            stats = {'Loss/total': loss.item(),
                     'Loss/GIoU': loss_giou.item(),
                     'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                     'Loss/clf_loss_test': clf_loss_test.item(),
                     'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                     'mIoU': ious.mean().item(),
                     'maxIoU': ious.max().item(),
                     'minIoU': ious.min().item(),
                     'mIoU_pred_center': ious_pred_center.mean().item()}

            if ious.max().item() > 0:
                stats['stdIoU'] = ious[ious > 0].std().item()
            # print(loss, stats)
            # print(torch.cuda.memory_summary())

        return loss, stats


class MyToMPActorParallel(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None, multi_gpu=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.multi_gpu = multi_gpu

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    # we can try three things fir initial exp:
    # 1 send in the prev fram as third training image
    # 2 send in the trajectory encoding path
    # 3 send in the prev saliency positions pos encoding

    # loss based optimization: do all the loss computation in a loss.backward() loop at once
    def __call__(self, data, train_batch_i=0, save_data_to_file=lambda x, y: None, is_training=False):
        # debug = True
        debug = False

        def printv(x, y):
            print(f"{x}: {y.shape} , mean:{y.mean():.4f}, min:{y.min():.4f}, max:{y.max():.4f}")

        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        num_frames = data['test_images'].shape[0]
        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_bb = data['train_anno']
        train_label = data['train_label']
        train_ltrb_target = data['train_ltrb_target']
        prev_features = None
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Classification features
        train_feat_head = self.net.get_backbone_head_feat(
            self.net.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:])))
        test_feat_head = self.net.get_backbone_head_feat(
            self.net.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:])))
        num_sequences = train_bb.shape[1]
        # del train_imgs
        # del test_imgs
        if debug:
            # torch.Size([4, 1024, 18, 18])
            printv("train_feat_head", train_feat_head)
            printv("test_feat_head", test_feat_head)

        train_feat = self.net.head.extract_head_feat(train_feat_head, num_sequences)
        test_feat = self.net.head.extract_head_feat(test_feat_head, num_sequences)
        if debug:
            # torch.Size([2, 2, 256, 18, 18])
            printv("train_feat", train_feat)
            printv("test_feat", test_feat)
            printv("train_ltrb_target", train_ltrb_target)

        # exit(0)
        # test_feat_list = test_feat.split(dim=0)
        # loss_all = []
        stats = {}
        loss = None
        iou_list = []

        output_tscores = []
        output_bbpreds = []
        all_stats = []
        num_frames_back_prop = 20
        for i in range(num_frames):
            # print(f"{i} frame running")
            if loss is not None and i % num_frames_back_prop == 0 and is_training:
                # print("Running backward")
                #  This makes sure graph is utilized and deleted
                # but retain graph is set to true so that the subgraphs used for {}_feat is not deleted
                (loss * (1 / num_frames_back_prop)).backward(retain_graph=True)
                # this does two jobs.
                # 1: we dont need to worry about the loss for whihc the grad is already computed
                # 2 the majority of the graph except the subgraphs for {}_feat is deleted
                loss = 0

            if loss is None:
                loss = 0

            # target_scores, bbox_preds = self.net(train_imgs,
            #                                      test_imgs,
            #                                      train_bb,
            #                                      train_label=train_label,
            #                                      train_ltrb_target=train_ltrb_target,
            #                                      prev_features=prev_features)
            x = False
            if x:
                cls_filter, breg_filter, test_feat_enc = self.net.head.get_filter_and_features(train_feat,
                                                                                               test_feat[i][None],
                                                                                               train_label=train_label,
                                                                                               train_ltrb_target=train_ltrb_target,
                                                                                               prev_features=prev_features)

                target_scores = self.net.head.classifier(test_feat_enc, cls_filter)
                bbox_preds = self.net.head.bb_regressor(test_feat_enc, breg_filter)
            else:
                # UNNECESSARY CONVOLUTIONS INSIDE LOOP
                # print("head is multi gpu: ",is_multi_gpu(self.net.head))
                target_scores, bbox_preds = self.net.head(train_feat, test_feat[i][None], train_bb,
                                                          train_label=train_label,
                                                          train_ltrb_target=train_ltrb_target,
                                                          prev_features=prev_features,
                                                          extract_head_feat=False)

            loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'][i][None],
                                                     data['test_sample_region'][i][None])

            # Classification losses for the different optimization iterations
            clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'][i][None],
                                                       data['test_anno'][i][None])

            loss += self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test
            # loss_all.append(loss)
            if torch.isnan(loss):
                raise ValueError('NaN detected in loss')

            ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'][i][None],
                                                                 bbox_preds)

            stats_i = {'Loss/total': loss.item(),
                       'Loss/GIoU': loss_giou.item(),
                       'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                       'Loss/clf_loss_test': clf_loss_test.item(),
                       'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                       'mIoU': ious.mean().item(),
                       'maxIoU': ious.max().item(),
                       'minIoU': ious.min().item(),
                       'mIoU_pred_center': ious_pred_center.mean().item()}

            if ious.max().item() > 0:
                stats_i['stdIoU'] = ious[ious > 0].std().item()
            # print(loss, stats)
            stats = {k: stats_i.get(k, 0) + stats.get(k, 0) for k in set(stats_i) | set(stats)}

            output_tscores.append(target_scores.detach())
            output_bbpreds.append(bbox_preds.detach())
            iou_list.append(ious_pred_center.detach())
            all_stats.append(stats_i)
            # print(torch.cuda.memory_summary())
        stats = {k: stats.get(k, 0) / num_frames for k in set(stats)}
        get_iou_disp_char(iou_list, num_frames, data['test_images'].shape[1])
        if train_batch_i % 50 == 0:
            save_data_to_file({'data': data, 'output_bbpreds': output_bbpreds, 'output_tscores': output_tscores,
                               'iou_list': iou_list, 'all_stats': all_stats}, f'{train_batch_i}-inp-op')
        return loss, stats


def get_iou_disp_char(ious, num_frames, batch_size):
    stri = '\n'
    for bi in range(batch_size):
        for fi in range(num_frames):
            for ch, v in zip(reversed(iou_char), reversed(iou_range_split)):
                # print(v)
                if ious[fi][bi] >= v:
                    stri += ch
                    break
        stri += '\n'
    print(stri)


# def save_data_to_file(self, data, i):
#     dir_path = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
#     dir_path = os.path.join(dir_path, 'data_inp/')
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#     file_path = os.path.join(dir_path, f'data_{i}_all.pt')
#     torch.save(data, file_path)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TrajPOCActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None, multi_gpu=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.multi_gpu = multi_gpu

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def predict_analytically(self, cur_img, prev_img, traj_grid):
        # print('dsadads', (cur_img.sum(dim=1)>0).nonzero())
        # print(traj.shape)
        h, w = cur_img.shape[-2:]
        map_pts = []
        for i in range(cur_img.shape[0]):
            pts = (cur_img[i, ...].sum(dim=0) > 0).nonzero()
            print('nonzero pts', pts)
            norm_t = [traj_grid[i, ..., pt[0], pt[1]] for pt in pts]
            new_pts = [(int(t[0] * h + pt[0]), int(pt[1] + t[1] * w)) for t, pt in zip(norm_t, pts)]
            # print([prev_img[i,...,pt[0], pt[1]] for pt in new_pts])
            map_pts.append(pts[next((pti for pti, pt in enumerate(new_pts) if prev_img[i, 1, pt[0], pt[1]] > 0), -1)])
        print(map_pts)
        return map_pts

    def accuracy_metric(self, scores, labels, cur_img=None, prev_img=None):
        # print(scores.shape, labels.shape)
        if cur_img is not None and prev_img is not None:
            import matplotlib.pyplot as plt
            plt.figure()
            for i in range(scores.shape[0]):
                score = scores[i, ...].detach().numpy()
                label = labels[i, ...].detach().numpy()
                plt.imshow(cur_img[i, ...].permute(1, 2, 0).detach().numpy())
                plt.show()
                plt.imshow(prev_img[i, ...].permute(1, 2, 0).detach().numpy())
                plt.show()
                plt.imshow(label)
                plt.show()
                plt.imshow(score)
                plt.show()

        l = labels.view(labels.shape[0], -1)
        s = scores.view(scores.shape[0], -1)
        # print(l.max(dim=1).indices)
        # print(s.max(dim=1).indices)
        return (s.max(dim=1).indices == l.max(dim=1).indices).sum() / labels.shape[0]
        # print((l==torch.max(l, dim=1)).nonzero())
        # return 0

    # we can try three things fir initial exp:
    # 1 send in the prev fram as third training image
    # 2 send in the trajectory encoding path
    # 3 send in the prev saliency positions pos encoding

    # loss based optimization: do all the loss computation in a loss.backward() loop at once
    def __call__(self, data, train_batch_i=0, save_data_to_file=lambda x, y: None, is_training=False, num_gpu=None):
        # debug = True
        debug = False

        def printv(x, y):
            print(f"{x}: {y.shape} , mean:{y.mean():.4f}, min:{y.min():.4f}, max:{y.max():.4f}")

        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # num_frames = data['test_images'].shape[0]
        current_image = data['current_image']
        previous_image = data['previous_image']
        trajectory = data['trajectory']
        label = data['label']
        # train_ltrb_target = data['train_ltrb_target']
        # prev_features = None
        # assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        if False:
            printv('current_image', current_image)
            printv('previous_image', previous_image)
            printv('trajectory', trajectory)
            printv('label', label)
        # current_image: torch.Size([1, 50, 3, 18, 18]), mean: 0.0062, min: 0.0000, max: 1.0000
        # previous_image: torch.Size([1, 50, 3, 18, 18]), mean: 0.0072, min: 0.0000, max: 1.0000
        # trajectory: torch.Size([1, 50, 2, 18, 18]), mean: -0.0004, min: -0.2778, max: 0.2222
        # label: torch.Size([1, 50, 18, 18]), mean: 0.0029, min: 0.0000, max: 0.1965

        target_scores = self.net(cur_img=current_image, prev_img=previous_image, trajectory=trajectory,
                                 train_label=None, train_ltrb_target=None)

        # loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])
        # print('target_scores.shape', target_scores.squeeze(0).shape)
        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, label, None)

        loss = self.loss_weight['test_clf'] * clf_loss_test

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        # ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)
        accuracy = self.accuracy_metric(target_scores.squeeze(0),
                                        label.squeeze(0))  # , current_image.squeeze(0), previous_image.squeeze(0))
        # self.predict_analytically( current_image.squeeze(0), previous_image.squeeze(0), trajectory.squeeze(0))
        stats = {'Loss/total': loss.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                 'accuracy': accuracy}

        return loss, stats


class MyToMPActorTempExtended(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None, multi_gpu=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.multi_gpu = multi_gpu

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __extract_temporal_data__(self, feature, label, ltrb, ground_label):
        # print('feature.shape, label.shape, ltrb.shape, ground_label.shape')
        # print(feature.shape, label.shape, ltrb.shape, ground_label.shape)
        feature = feature.permute(0, 1, 3, 4, 2)
        ltrb = ltrb.permute(0, 1, 3, 4, 2)

        mask = ground_label > 0.01
        return {'select_feat': feature[mask],
                'select_label': label[mask],
                'select_ltrb': ltrb[mask],
                'mask': mask}

    # def __pack_temporal_data__(self, temporal_data_dicts):
    #     if len(temporal_data_dicts) == 0:
    #         return None
    #     mask = torch.cat([di['mask'] for di in temporal_data_dicts], dim=0)
    #     feat = torch.cat([di['select_feat'] for di in temporal_data_dicts], dim=0)
    #     label = torch.cat([di['select_label'] for di in temporal_data_dicts], dim=0)
    #     ltrb = torch.cat([di['select_ltrb'] for di in temporal_data_dicts], dim=0)
    #     # print(mask.shape, feat.shape, label.shape, ltrb.shape)
    #     return {'mask': mask, 'feat': feat, 'label': label, 'ltrb': ltrb}

    def __pack_temporal_data__(self, test_feat, pred_label, pred_ltrb, pred_mask, i, num_frames_carry):
        # torch.Size([50, 1, 256, 18, 18]) torch.Size([50, 1, 18, 18]) torch.Size([50, 1, 4, 18, 18]) torch.Size([50, 1, 18, 18]) 1 here is batch size
        # print(test_feat.shape, pred_label.shape, pred_ltrb.shape, pred_mask.shape)
        if i == 0:
            return None
        start = max(0, i - num_frames_carry)
        end = i
        # m = pred_mask[start:end]
        # feats = []
        # labels = []
        # masks = []
        # for b_i in range(test_feat.shape[1]):
        #     feats.append(test_feat[start:end, b_i, ...].permute(0, 2, 3, 1)[m[:, b_i, ...]])
        #     labels.append(pred_label[start:end, b_i, ...][m[:, b_i, ...]])
        #     masks.append(m[:,b_i, ...])
        return {
            'feats': test_feat[start:end],
            'labels': pred_label[start:end],
            'ltrb': pred_ltrb[start:end],
            'mask': pred_mask[start:end]
        }

    # we can try three things fir initial exp:
    # 1 send in the prev fram as third training image
    # 2 send in the trajectory encoding path
    # 3 send in the prev saliency positions pos encoding

    # loss based optimization: do all the loss computation in a loss.backward() loop at once
    def __call__(self, data, train_batch_i=0, save_data_to_file=lambda x, y: None, is_training=False):
        # debug = True
        debug = False

        def printv(x, y):
            if y.dtype != torch.bool:
                print(
                    f"{x}: {y.shape} , mean:{y.mean():.4f}, min:{y.min():.4f}, max:{y.max():.4f}, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")
            else:
                print(
                    f"{x}: {y.shape} , mean:-----, min:-----, max:-----, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")

        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_bb = data['train_anno']
        train_label = data['train_label']
        train_ltrb_target = data['train_ltrb_target']

        test_label = data['test_label']
        test_ltrb_target = data['test_ltrb_target']

        prev_features = None
        num_frames = test_imgs.shape[0]

        pred_label = torch.zeros(test_label.shape, dtype=test_label.dtype)
        pred_ltrb = torch.zeros(test_ltrb_target.shape, dtype=test_ltrb_target.dtype)
        pred_mask = torch.zeros(test_label.shape, dtype=torch.bool)

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Classification features
        train_feat_head = self.net.get_backbone_head_feat(
            self.net.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:])))
        test_feat_head = self.net.get_backbone_head_feat(
            self.net.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:])))
        num_sequences = train_bb.shape[1]
        # del train_imgs
        # del test_imgs
        if debug:
            # torch.Size([4, 1024, 18, 18])
            printv("train_feat_head", train_feat_head)
            printv("test_feat_head", test_feat_head)

        train_feat = self.net.head.extract_head_feat(train_feat_head, num_sequences)
        test_feat = self.net.head.extract_head_feat(test_feat_head, num_sequences)
        if debug:
            # torch.Size([2, 2, 256, 18, 18])
            printv("train_feat", train_feat)
            printv("test_feat", test_feat)
            printv('train_feat', train_feat)
            printv("train_ltrb_target", train_ltrb_target)
            printv('test_label', test_label)
            printv("test_ltrb_target", test_ltrb_target)

        # exit(0)
        # test_feat_list = test_feat.split(dim=0)
        # loss_all = []
        stats = {}
        loss = None
        iou_list = []
        #
        # output_tscores = []
        # output_bbpreds = []
        all_stats = []
        num_frames_back_prop = 10
        num_frames_carry = 10
        for i in range(num_frames):
            if debug:
                print(f"{i} frame running")
            if loss is not None and i % num_frames_back_prop == 0 and is_training:
                # print("Running backward")
                #  This makes sure graph is utilized and deleted
                # but retain graph is set to true so that the subgraphs used for {}_feat is not deleted
                (loss * (1 / num_frames_back_prop)).backward(retain_graph=True)
                # this does two jobs.
                # 1: we dont need to worry about the loss for whihc the grad is already computed
                # 2 the majority of the graph except the subgraphs for {}_feat is deleted
                loss = 0

            if loss is None:
                loss = 0

            # target_scores, bbox_preds = self.net(train_imgs,
            #                                      test_imgs,
            #                                      train_bb,
            #                                      train_label=train_label,
            #                                      train_ltrb_target=train_ltrb_target,
            #                                      prev_features=prev_features)
            temporal_features = self.__pack_temporal_data__(test_feat, pred_label, pred_ltrb, pred_mask, i,
                                                            num_frames_carry)
            if debug and temporal_features is not None:
                for k in temporal_features.keys():
                    printv(k, temporal_features[k])

            x = False
            if x:
                cls_filter, breg_filter, test_feat_enc = self.net.head.get_filter_and_features(train_feat,
                                                                                               test_feat[i][None],
                                                                                               train_label=train_label,
                                                                                               train_ltrb_target=train_ltrb_target,
                                                                                               temporal_features=temporal_features)

                target_scores = self.net.head.classifier(test_feat_enc, cls_filter)
                bbox_preds = self.net.head.bb_regressor(test_feat_enc, breg_filter)
            else:
                # UNNECESSARY CONVOLUTIONS INSIDE LOOP
                # print("head is multi gpu: ",is_multi_gpu(self.net.head))
                target_scores, bbox_preds = self.net.head(train_feat, test_feat[i][None], train_bb,
                                                          train_label=train_label,
                                                          train_ltrb_target=train_ltrb_target,
                                                          temporal_features=temporal_features,
                                                          extract_head_feat=False)

            # carrying_data.append(self.__extract_temporal_data__(test_feat[i][None], target_scores, bbox_preds,
            #                                                     data['test_label'][i][None]))

            pred_label[i, ...] = target_scores[0, ...].clone().detach()
            pred_ltrb[i, ...] = bbox_preds[0, ...].clone().detach()
            # SEND PREDICTED SCORES OR THE GROUND TRUTH SCORES
            self.__update_mask__(target_scores, pred_mask[i, ...])

            # print("Predicted:", target_scores.shape, bbox_preds.shape)
            # print("label", data['test_label'][i])
            loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'][i][None],
                                                     data['test_sample_region'][i][None])

            # Classification losses for the different optimization iterations
            clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'][i][None],
                                                       data['test_anno'][i][None])

            loss += self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test
            # loss_all.append(loss)
            if torch.isnan(loss):
                raise ValueError('NaN detected in loss')

            ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'][i][None],
                                                                 bbox_preds)

            stats_i = {'Loss/total': loss.item(),
                       'Loss/GIoU': loss_giou.item(),
                       'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                       'Loss/clf_loss_test': clf_loss_test.item(),
                       'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                       'mIoU': ious.mean().item(),
                       'maxIoU': ious.max().item(),
                       'minIoU': ious.min().item(),
                       'mIoU_pred_center': ious_pred_center.mean().item()}

            if ious.max().item() > 0:
                stats_i['stdIoU'] = ious[ious > 0].std().item()
            # print(loss, stats)
            stats = {k: stats_i.get(k, 0) + stats.get(k, 0) for k in set(stats_i) | set(stats)}

            # output_tscores.append(target_scores.detach())
            # output_bbpreds.append(bbox_preds.detach())
            iou_list.append(ious_pred_center.detach())
            all_stats.append(stats_i)
            # print(torch.cuda.memory_summary())
        stats = {k: stats.get(k, 0) / num_frames for k in set(stats)}
        get_iou_disp_char(iou_list, num_frames, data['test_images'].shape[1])
        # if train_batch_i % 500 == 0:
        #     save_data_to_file({'data': data, 'iou_list': iou_list, 'all_stats': all_stats}, f'{train_batch_i}-inp-op')
        return loss, stats

    def __update_mask__(self, target_scores, pred_mask, num_pass_on_feat=25):
        for i in range(target_scores.shape[1]):
            pred_mask[i, ...].view(-1)[
                torch.maximum(target_scores[0, i, ...], torch.tensor(1.0e-5)).view(1, -1).multinomial(
                    num_pass_on_feat)] = True


# =======================================================================================================================================================================================================
# =======================================================================================================================================================================================================


class ParallelMyToMPActorTempExtended(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None, multi_gpu=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.multi_gpu = multi_gpu

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        h, w = scores.shape[-2:]
        # print(scores.shape, ltrb_gth.shape, ltrb_pred.shape)
        scores = scores.view(1, -1, h, w)
        ltrb_gth = ltrb_gth.view(1, -1, 4, h, w)
        ltrb_pred = ltrb_pred.view(1, -1, 4, h, w)
        # print(scores.shape, ltrb_gth.shape, ltrb_pred.shape)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    # we can try three things fir initial exp:
    # 1 send in the prev fram as third training image
    # 2 send in the trajectory encoding path
    # 3 send in the prev saliency positions pos encoding
    # loss based optimization: do all the loss computation in a loss.backward() loop at once
    def __call__(self, data, train_batch_i=0, save_data_to_file=lambda x, y: None, is_training=False, num_gpu=None):
        # debug = True
        debug = False

        def printv(x, y):
            if y.dtype != torch.bool:
                print(
                    f"{x}: {y.shape} , mean:{y.mean():.4f}, min:{y.min():.4f}, max:{y.max():.4f}, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")
            else:
                print(
                    f"{x}: {y.shape} , mean:-----, min:-----, max:-----, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        num_frames_carry = 10

        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_bb = data['train_anno']
        train_label = data['train_label']
        train_ltrb_target = data['train_ltrb_target']

        test_label = data['test_label']
        test_ltrb_target = data['test_ltrb_target']

        batch_sz = test_imgs.shape[1]
        num_frames = test_imgs.shape[0]

        if debug:
            printv('train_imgs', train_imgs)
            printv('test_imgs', test_imgs)

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        # Classification features
        train_feat_head = self.net.get_backbone_head_feat(
            self.net.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))).unsqueeze(0)
        test_feat_head = self.net.get_backbone_head_feat(
            self.net.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))).unsqueeze(0)
        # num_test_sequences = test_imgs.shape[1]
        # del train_imgs
        # del test_imgs
        if debug:
            # torch.Size([4, 1024, 18, 18])
            printv("train_feat_head", train_feat_head)
            printv("test_feat_head", test_feat_head)
            printv("train_bb", train_bb)
            print("\n")

        # train_feat = self.net.head.extract_head_feat(train_feat_head, num_sequences)
        # test_feat = self.net.head.extract_head_feat(test_feat_head, num_sequences)
        train_feat, test_feat = self.net.head(train_feat_head, test_feat_head, batch_sz='for_multi_gpu',
                                              extract_head_feat_only=True, temporal_features=None)
        train_feat = train_feat.view(*train_label.shape[:2], *train_feat.shape[-3:])
        test_feat = test_feat.view(*test_label.shape[:2], *test_feat.shape[-3:])

        test_feat_untampered = test_feat
        if debug:
            # torch.Size([2, 2, 256, 18, 18])
            printv("train_feat", train_feat)
            printv("test_feat", test_feat)
            printv('train_label', train_label)
            printv("train_ltrb_target", train_ltrb_target)
            printv('test_label', test_label)
            printv("test_ltrb_target", test_ltrb_target)

        # expand train images according to test seq length
        train_feat = train_feat.unsqueeze(1).repeat(1, num_frames, 1, 1, 1, 1)  # tr x te x B x C x h x w
        train_ltrb_target = train_ltrb_target.unsqueeze(1).repeat(1, num_frames, 1, 1, 1, 1)  # tr x te x B x 4 x h x w
        train_label = train_label.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)  # tr x te x B x h x w
        test_feat = test_feat.unsqueeze(0)  # 1  x te x B x C x h x w

        # mix the test frames into batch dimension
        # tr x te x B x C x h x w => tr x (te*B) x C x h x w
        train_feat = train_feat.flatten(start_dim=1, end_dim=2)
        train_ltrb_target = train_ltrb_target.flatten(start_dim=1, end_dim=2)
        train_label = train_label.flatten(start_dim=1, end_dim=2)
        test_feat = test_feat.flatten(start_dim=1, end_dim=2)

        if debug:
            print("modded: train_feat", train_feat.shape)
            print("modded: train_ltrb_target", train_ltrb_target.shape)
            print("modded: train_label", train_label.shape)
            print("modded: test_feat", test_feat.shape)
            print("test_feat_untampered", test_feat_untampered.shape)

        with torch.no_grad():
            pred_label, pred_ltrb = self.net.head(train_feat, test_feat, train_bb=None,
                                                  train_label=train_label,
                                                  train_ltrb_target=train_ltrb_target,
                                                  temporal_features=None,
                                                  extract_head_feat=False)
        if debug:
            printv('pred_label', pred_label)
            printv('pred_ltrb', pred_ltrb)

        pred_label = pred_label.view(num_frames, batch_sz, *train_feat.shape[-2:]).detach()
        pred_ltrb = pred_ltrb.view(num_frames, batch_sz, 4, *train_feat.shape[-2:]).detach()

        if debug:
            printv('pred_label', pred_label)
            printv('pred_ltrb', pred_ltrb)

        topkindices = torch.topk(torch.maximum(pred_label.flatten(2), torch.tensor(1e-5)), k=25, dim=2).indices
        pred_mask = torch.zeros(pred_label.shape, dtype=torch.bool, device=pred_label.device).flatten(2).scatter_(2,
                                                                                                                  topkindices,
                                                                                                                  True).view(
            *pred_label.shape)

        # [50, 2, 256, 18, 18] => [10+49, 2, 256, 18, 18]
        expand_to_frames = [0] * num_frames_carry + list(range(num_frames - 1))
        temp_feat = test_feat_untampered[expand_to_frames].detach()
        temp_feat = temp_feat.unfold(0, num_frames_carry, 1).permute(5, 0, 1, 2, 3, 4)
        temp_feat = temp_feat.flatten(start_dim=1, end_dim=2)

        temp_ltrb = pred_ltrb[expand_to_frames]
        temp_ltrb = temp_ltrb.unfold(0, num_frames_carry, 1).permute(5, 0, 1, 2, 3, 4)
        temp_ltrb = temp_ltrb.flatten(start_dim=1, end_dim=2)

        temp_label = pred_label[expand_to_frames]
        temp_label = temp_label.unfold(0, num_frames_carry, 1).permute(4, 0, 1, 2, 3)
        temp_label = temp_label.flatten(start_dim=1, end_dim=2)

        temp_mask = pred_mask[expand_to_frames]
        temp_mask = temp_mask.unfold(0, num_frames_carry, 1).permute(4, 0, 1, 2, 3)
        temp_mask = temp_mask.flatten(start_dim=1, end_dim=2)

        if debug:
            printv("Temporal: temp_feat", temp_feat)
            printv("Temporal: temp_label", temp_label)
            printv("Temporal: temp_ltrb", temp_ltrb)
            printv("Temporal: temp_mask", temp_mask)

        temporal_features = TensorDict({
            'feats': temp_feat,
            'labels': temp_label,
            'ltrb': temp_ltrb,
            'mask': temp_mask
        })

        target_scores, bbox_preds = self.net.head(train_feat, test_feat, train_bb=None,
                                                  train_label=train_label,
                                                  train_ltrb_target=train_ltrb_target,
                                                  temporal_features=temporal_features,
                                                  extract_head_feat=False)
        target_scores = target_scores.view(test_label.shape)
        bbox_preds = bbox_preds.view(test_ltrb_target.shape)

        if debug:
            printv('Result: target_scores', target_scores)
            printv('Result: bbox_preds', bbox_preds)
            printv('gt: bbox_preds', test_ltrb_target)

        loss_giou, ious = self.objective['giou'](bbox_preds, test_ltrb_target, data['test_sample_region'])

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, test_ltrb_target, bbox_preds)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious > 0].std().item()
        get_iou_disp_char(ious_pred_center.view(*test_label.shape[:2]), num_frames, batch_sz)
        return loss, stats


# ===================================================================================================
# ===================================================================================================
# ===================================================================================================


class MoreParallelMyToMPActorTempExtended(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None, multi_gpu=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.multi_gpu = multi_gpu

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        h, w = scores.shape[-2:]
        # print(scores.shape, ltrb_gth.shape, ltrb_pred.shape)
        scores = scores.view(1, -1, h, w)
        ltrb_gth = ltrb_gth.view(1, -1, 4, h, w)
        ltrb_pred = ltrb_pred.view(1, -1, 4, h, w)
        # print(scores.shape, ltrb_gth.shape, ltrb_pred.shape)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    # we can try three things fir initial exp:
    # 1 send in the prev fram as third training image
    # 2 send in the trajectory encoding path
    # 3 send in the prev saliency positions pos encoding

    # loss based optimization: do all the loss computation in a loss.backward() loop at once
    def __call__(self, data, train_batch_i=0, save_data_to_file=lambda x, y: None, is_training=False, num_gpu=None):
        debug = True

        # debug = False

        def printv(x, y):
            if y.dtype != torch.bool:
                print(
                    f"{x}: {y.shape} , mean:{y.mean():.4f}, min:{y.min():.4f}, max:{y.max():.4f}, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")
            else:
                print(
                    f"{x}: {y.shape} , mean:-----, min:-----, max:-----, contiguous: {y.is_contiguous()}, has_grad: {y.requires_grad, y.grad_fn}")

        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        num_frames_carry = 10

        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_bb = data['train_anno']
        train_label = data['train_label']
        train_ltrb_target = data['train_ltrb_target']

        test_label = data['test_label']
        test_ltrb_target = data['test_ltrb_target']
        test_sample_region = data['test_sample_region']
        test_anno = data['test_anno']

        batch_sz = test_imgs.shape[1]
        num_frames = test_imgs.shape[0]

        if debug:
            printv('train_imgs', train_imgs)
            printv('test_imgs', test_imgs)

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        # Classification features
        train_feat_head = self.net.get_backbone_head_feat(
            self.net.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))).unsqueeze(0)
        test_feat_head = self.net.get_backbone_head_feat(
            self.net.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))).unsqueeze(0)
        # num_test_sequences = test_imgs.shape[1]
        # del train_imgs
        # del test_imgs
        if debug:
            # torch.Size([4, 1024, 18, 18])
            printv("train_feat_head", train_feat_head)
            printv("test_feat_head", test_feat_head)
            printv("train_bb", train_bb)
            print("\n")

        # train_feat = self.net.head.extract_head_feat(train_feat_head, num_sequences)
        # test_feat = self.net.head.extract_head_feat(test_feat_head, num_sequences)
        train_feat, test_feat = self.net.head(train_feat_head, test_feat_head, batch_sz='for_multi_gpu',
                                              extract_head_feat_only=True, temporal_features=None)
        train_feat = train_feat.view(*train_label.shape[:2], *train_feat.shape[-3:])
        test_feat = test_feat.view(*test_label.shape[:2], *test_feat.shape[-3:])

        test_feat_untampered = test_feat
        if debug:
            # torch.Size([2, 2, 256, 18, 18])
            printv("train_feat", train_feat)
            printv("test_feat", test_feat)
            printv('train_label', train_label)
            printv("train_ltrb_target", train_ltrb_target)
            printv('test_label', test_label)
            printv("test_ltrb_target", test_ltrb_target)

        # expand train images according to test seq length
        train_feat = train_feat.unsqueeze(1).repeat(1, num_frames, 1, 1, 1, 1)  # tr x te x B x C x h x w
        train_ltrb_target = train_ltrb_target.unsqueeze(1).repeat(1, num_frames, 1, 1, 1, 1)  # tr x te x B x 4 x h x w
        train_label = train_label.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)  # tr x te x B x h x w
        test_feat = test_feat.unsqueeze(0)  # 1  x te x B x C x h x w

        # mix the test frames into batch dimension
        # tr x te x B x C x h x w => tr x (te*B) x C x h x w
        train_feat = train_feat.flatten(start_dim=1, end_dim=2)
        train_ltrb_target = train_ltrb_target.flatten(start_dim=1, end_dim=2)
        train_label = train_label.flatten(start_dim=1, end_dim=2)
        test_feat = test_feat.flatten(start_dim=1, end_dim=2)

        if debug:
            print("modded: train_feat", train_feat.shape)
            print("modded: train_ltrb_target", train_ltrb_target.shape)
            print("modded: train_label", train_label.shape)
            print("modded: test_feat", test_feat.shape)
            print("test_feat_untampered", test_feat_untampered.shape)

        split_sz = 1 * num_gpu
        split_ = lambda in_tensor: in_tensor.split(split_sz, dim=1)
        with torch.no_grad():
            pred_label = []
            pred_ltrb = []
            for s_tr_ft, s_te_ft, s_tr_lb, s_tr_ltrb in zip(split_(train_feat), split_(test_feat), split_(train_label),
                                                            split_(train_ltrb_target)):
                # printv('s_tr_ft', s_tr_ft)
                # printv('s_te_ft', s_te_ft)
                # printv('s_tr_lb', s_tr_lb)
                # printv('s_tr_ltrb', s_tr_ltrb)
                s_pred_label, s_pred_ltrb = self.net.head(s_tr_ft, s_te_ft, train_bb=None,
                                                          train_label=s_tr_lb,
                                                          train_ltrb_target=s_tr_ltrb,
                                                          temporal_features=None,
                                                          extract_head_feat=False)
                pred_label.append(s_pred_label)
                pred_ltrb.append(s_pred_ltrb)
            # print([x.shape for x in pred_label])
            # print([x.shape for x in pred_ltrb])
            pred_label = torch.cat(pred_label, dim=1)
            pred_ltrb = torch.cat(pred_ltrb, dim=1)

        if debug:
            printv('pred_label', pred_label)
            printv('pred_ltrb', pred_ltrb)

        pred_label = pred_label.view(num_frames, batch_sz, *train_feat.shape[-2:]).detach()
        pred_ltrb = pred_ltrb.view(num_frames, batch_sz, 4, *train_feat.shape[-2:]).detach()

        if debug:
            printv('pred_label', pred_label)
            printv('pred_ltrb', pred_ltrb)

        topkindices = torch.topk(torch.maximum(pred_label.flatten(2), torch.tensor(1e-5)), k=25, dim=2).indices
        pred_mask = torch.zeros(pred_label.shape, dtype=torch.bool, device=pred_label.device).flatten(2).scatter_(2,
                                                                                                                  topkindices,
                                                                                                                  True).view(
            *pred_label.shape)

        # [50, 2, 256, 18, 18] => [10+49, 2, 256, 18, 18]
        expand_to_frames = [0] * num_frames_carry + list(range(num_frames - 1))
        temp_feat = test_feat_untampered[expand_to_frames].detach()
        temp_feat = temp_feat.unfold(0, num_frames_carry, 1).permute(5, 0, 1, 2, 3, 4)
        temp_feat = temp_feat.flatten(start_dim=1, end_dim=2)

        temp_ltrb = pred_ltrb[expand_to_frames]
        temp_ltrb = temp_ltrb.unfold(0, num_frames_carry, 1).permute(5, 0, 1, 2, 3, 4)
        temp_ltrb = temp_ltrb.flatten(start_dim=1, end_dim=2)

        temp_label = pred_label[expand_to_frames]
        temp_label = temp_label.unfold(0, num_frames_carry, 1).permute(4, 0, 1, 2, 3)
        temp_label = temp_label.flatten(start_dim=1, end_dim=2)

        temp_mask = pred_mask[expand_to_frames]
        temp_mask = temp_mask.unfold(0, num_frames_carry, 1).permute(4, 0, 1, 2, 3)
        temp_mask = temp_mask.flatten(start_dim=1, end_dim=2)

        if debug:
            printv("Temporal: temp_feat", temp_feat)
            printv("Temporal: temp_label", temp_label)
            printv("Temporal: temp_ltrb", temp_ltrb)
            printv("Temporal: temp_mask", temp_mask)
        overall_loss = 0
        overall_loss_giou = 0
        overall_clf_loss_test = 0
        ious_list = []
        iou_center_pred_list = []
        for s_tr_ft, s_te_ft, s_tr_lb, s_tr_ltrb, s_ti_ft, s_ti_lb, s_ti_ltrb, s_ti_mk, s_te_ltrb, s_te_sam_re, s_te_lb, s_te_anno in zip(
                *[split_(in_te) for in_te in
                  [train_feat, test_feat, train_label, train_ltrb_target,
                   temp_feat, temp_label, temp_ltrb, temp_mask,
                   test_ltrb_target.view(*test_feat.shape[:2], *test_ltrb_target.shape[2:]),
                   test_sample_region.view(*test_feat.shape[:2], *test_sample_region.shape[2:]),
                   test_label.view(*test_feat.shape[:2], *test_label.shape[2:]),
                   test_anno.view(*test_feat.shape[:2], *test_anno.shape[2:])
                   ]]):
            temporal_features = TensorDict({
                'feats': s_ti_ft,
                'labels': s_ti_lb,
                'ltrb': s_ti_ltrb,
                'mask': s_ti_mk
            })
            s_target_scores, s_bbox_preds = self.net.head(s_tr_ft, s_te_ft, train_bb=None,
                                                          train_label=s_tr_lb,
                                                          train_ltrb_target=s_tr_ltrb,
                                                          temporal_features=temporal_features,
                                                          extract_head_feat=False)
            # target_scores = target_scores.view(test_label.shape)
            # bbox_preds = bbox_preds.view(test_ltrb_target.shape)
            if debug:
                printv('Result: target_scores', s_target_scores)
                printv('Result: bbox_preds', s_bbox_preds)
                printv('gt: bbox_preds', s_te_ltrb)

            loss_giou, ious = self.objective['giou'](s_bbox_preds, s_te_ltrb, s_te_sam_re)

            # Classification losses for the different optimization iterations
            clf_loss_test = self.objective['test_clf'](s_target_scores, s_te_lb, s_te_anno)

            loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test

            if torch.isnan(loss):
                raise ValueError('NaN detected in loss')

            ious_pred_center = self.compute_iou_at_max_score_pos(s_target_scores, s_te_ltrb, s_bbox_preds)
            loss.backward(retain_graph=True)
            print("Ran Loss Backward")
            overall_loss += loss.detach()
            overall_loss_giou += loss_giou.detach()
            overall_clf_loss_test += clf_loss_test.detach()
            ious_list.append(ious.detach())
            iou_center_pred_list.append(ious_pred_center.detach())
            del loss
            del s_target_scores
            del s_bbox_preds

        ious = torch.cat(ious_list, dim=1)
        ious_pred_center = torch.cat(iou_center_pred_list, dim=1)
        stats = {'Loss/total': overall_loss,
                 'Loss/GIoU': overall_loss_giou,
                 'Loss/weighted_GIoU': self.loss_weight['giou'] * overall_loss_giou,
                 'Loss/clf_loss_test': overall_clf_loss_test,
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * overall_clf_loss_test,
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious > 0].std().item()
        get_iou_disp_char(ious_pred_center.view(*test_label.shape[:2]), num_frames, batch_sz)
        return torch.zeros(1, requires_grad=True), stats
