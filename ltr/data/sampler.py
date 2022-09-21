import random
import torch.utils.data

from ltr.data.processing_utils import gauss_2d
from pytracking import TensorDict
import numpy as np


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames, used to learn the DiMP classification model and obtain the
    modulation vector for IoU-Net, and ii) a set of test frames on which target classification loss for the predicted
    DiMP model, and the IoU prediction loss for the IoU-Net is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """
        # print("Trying to get a sample")

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_train_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_test_frames,
                                                              min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible) - self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] + 1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        # print(train_frame_ids)
        # print(test_frame_ids)
        # exit(0)
        train_frames, train_anno, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        test_frames, test_anno, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name(),
                           'test_class': meta_obj_test.get('object_class_name')})

        return self.processing(data)


class DiMPSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)


class ATOMSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames=1, num_train_frames=1, processing=no_processing, frame_sample_mode='interval'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)


class LWLSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames and ii) a set of test frames. The train frames, along with the
    ground-truth masks, are passed to the few-shot learner to obtain the target model parameters \tau. The test frames
    are used to compute the prediction accuracy.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is randomly
    selected from that dataset. A base frame is then sampled randomly from the sequence. The 'train frames'
    are then sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id], and the 'test frames'
    are sampled from the sequence from the range (base_frame_id, base_frame_id + max_gap] respectively. Only the frames
    in which the target is visible are sampled. If enough visible frames are not found, the 'max_gap' is increased
    gradually until enough frames are found. Both the 'train frames' and the 'test frames' are sorted to preserve the
    temporal order.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, p_reverse=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            p_reverse - Probability that a sequence is temporally reversed
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing

        self.p_reverse = p_reverse

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (dataset index)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        reverse_sequence = False
        if self.p_reverse is not None:
            reverse_sequence = random.random() < self.p_reverse

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames)

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
            while test_frame_ids is None:
                if gap_increase > 1000:
                    raise Exception('Frame not found')

                if not reverse_sequence:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible) - self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] + 1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
                else:
                    # Sample in reverse order, i.e. train frames come after the test frames
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_test_frames + 1,
                                                             max_id=len(visible) - self.num_train_frames - 1)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0],
                                                              max_id=base_frame_id[0] + self.max_gap + gap_increase)
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=0,
                                                              max_id=train_frame_ids[0] - 1,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        # Sort frames
        train_frame_ids = sorted(train_frame_ids, reverse=reverse_sequence)
        test_frame_ids = sorted(test_frame_ids, reverse=reverse_sequence)

        all_frame_ids = train_frame_ids + test_frame_ids

        # Load frames
        all_frames, all_anno, meta_obj = dataset.get_frames(seq_id, all_frame_ids, seq_info_dict)

        train_frames = all_frames[:len(train_frame_ids)]
        test_frames = all_frames[len(train_frame_ids):]

        train_anno = {}
        test_anno = {}
        for key, value in all_anno.items():
            train_anno[key] = value[:len(train_frame_ids)]
            test_anno[key] = value[len(train_frame_ids):]

        train_masks = train_anno['mask'] if 'mask' in train_anno else None
        test_masks = test_anno['mask'] if 'mask' in test_anno else None

        data = TensorDict({'train_images': train_frames,
                           'train_masks': train_masks,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_masks': test_masks,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name()})

        return self.processing(data)


class KYSSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, sequence_sample_info, processing=no_processing,
                 sample_occluded_sequences=False):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            sequence_sample_info - A dict containing information about how to sample a sequence, e.g. number of frames,
                                    max gap between frames, etc.
            processing - An instance of Processing class which performs the necessary processing of the data.
            sample_occluded_sequences - If true, sub-sequence containing occlusion is sampled whenever possible
        """

        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.sequence_sample_info = sequence_sample_info
        self.processing = processing

        self.sample_occluded_sequences = sample_occluded_sequences

    def __len__(self):
        return self.samples_per_epoch

    # randomly choose #num_ids valid frames in the range
    def _sample_ids(self, valid, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(valid):
            max_id = len(valid)

        valid_ids = [i for i in range(min_id, max_id) if valid[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    # starting from frame id #first_occ_frame, find the id of next partially occluded frame
    def find_occlusion_end_frame(self, first_occ_frame, target_not_fully_visible):
        for i in range(first_occ_frame, len(target_not_fully_visible)):
            if not target_not_fully_visible[i]:
                return i

        return len(target_not_fully_visible)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        p_datasets = self.p_datasets

        dataset = random.choices(self.datasets, p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        num_train_frames = self.sequence_sample_info['num_train_frames']
        num_test_frames = self.sequence_sample_info['num_test_frames']
        max_train_gap = self.sequence_sample_info['max_train_gap']
        allow_missing_target = self.sequence_sample_info['allow_missing_target']
        min_fraction_valid_frames = self.sequence_sample_info.get('min_fraction_valid_frames', 0.0)

        if allow_missing_target:
            min_visible_frames = 0
        else:
            raise NotImplementedError

        valid_sequence = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not valid_sequence:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']
            visible_ratio = seq_info_dict.get('visible_ratio', visible)

            num_visible = visible.type(torch.int64).sum().item()

            # kalyan-understanding
            # we are only caring for video datasets (as below code not implemented for non video sequences)
            # so only the condition after or is evaluated
            enough_visible_frames = not is_video_dataset or (num_visible > min_visible_frames and len(visible) >= 20)

            valid_sequence = enough_visible_frames

        if self.sequence_sample_info['mode'] == 'Sequence':
            if is_video_dataset:
                train_frame_ids = None
                test_frame_ids = None
                gap_increase = 0

                test_valid_image = torch.zeros(num_test_frames, dtype=torch.int8)
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    occlusion_sampling = False
                    if dataset.has_occlusion_info() and self.sample_occluded_sequences:
                        target_not_fully_visible = visible_ratio < 0.9
                        if target_not_fully_visible.float().sum() > 0:
                            occlusion_sampling = True

                    if occlusion_sampling:
                        first_occ_frame = target_not_fully_visible.nonzero()[0]

                        occ_end_frame = self.find_occlusion_end_frame(first_occ_frame, target_not_fully_visible)

                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=max(0, first_occ_frame - 20),
                                                         max_id=first_occ_frame - 5)

                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)

                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids

                        end_frame = min(occ_end_frame + random.randint(5, 20), len(visible) - 1)

                        if (end_frame - base_frame_id) < num_test_frames:
                            rem_frames = num_test_frames - (end_frame - base_frame_id)
                            end_frame = random.randint(end_frame, min(len(visible) - 1, end_frame + rem_frames))
                            base_frame_id = max(0, end_frame - num_test_frames + 1)

                            end_frame = min(end_frame, len(visible) - 1)

                        step_len = float(end_frame - base_frame_id) / float(num_test_frames)

                        test_frame_ids = [base_frame_id + int(x * step_len) for x in range(0, num_test_frames)]
                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0] * (num_test_frames - len(test_frame_ids))
                    else:
                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=2 * num_train_frames,
                                                         max_id=len(visible) - int(
                                                             num_test_frames * min_fraction_valid_frames))
                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)
                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids

                        test_frame_ids = list(range(base_frame_id, min(len(visible), base_frame_id + num_test_frames)))
                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0] * (num_test_frames - len(test_frame_ids))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Get frames
        # print(dataset.get_name(), seq_id)
        # print(train_frame_ids)
        # print(test_frame_ids)
        train_frames, train_anno_dict, _ = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)
        test_anno = test_anno_dict['bbox']
        test_valid_anno = test_anno_dict['valid']
        test_visible = test_anno_dict['visible']
        test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))

        # Prepare data
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'test_valid_anno': test_valid_anno,
                           'test_visible': test_visible,
                           'test_valid_image': test_valid_image,
                           'test_visible_ratio': test_visible_ratio,
                           'dataset': dataset.get_name()})
        # exit(0)
        # Send for processing
        return self.processing(data)



class TrajectoryPOCSampler(torch.utils.data.Dataset):
    def __init__(self, im_shape, samples_per_epoch, traj_disp=None, num_points=6):
        self.im_shape = torch.tensor(im_shape)
        self.samples_per_epoch = samples_per_epoch
        self.num_points = num_points
        self.traj_disp = traj_disp
        if traj_disp is None:
            self.traj_disp = -((-self.im_shape[0] - self.im_shape[1]) // 10)  # ceil((h+w) // 10)

    def __len__(self):
        return self.samples_per_epoch

    def __is_in__(self, i, j):
        return 0 <= i < self.im_shape[0] and 0 <= j < self.im_shape[1]

    def __getitem__(self, index):
        unique_pts = []
        while len(unique_pts) < self.num_points:
            pt = ([random.randrange(0, self.im_shape[0]), random.randrange(0, self.im_shape[1])])
            if pt not in unique_pts:
                unique_pts.append(pt)
        mapped_pts = []
        for pt in unique_pts:
            new_pt = [random.randrange(max(0, pt[0] - self.traj_disp), min(pt[0] + self.traj_disp, self.im_shape[0])),
                      random.randrange(max(0, pt[1] - self.traj_disp), min(pt[1] + self.traj_disp, self.im_shape[1]))]
            mapped_pts.append(new_pt)

        duplicated_traj_code = False
        traj = [[n[0] - o[0], n[1] - o[1]]*(2 if duplicated_traj_code else 1) for (n, o) in zip(mapped_pts, unique_pts)]
        im_prev = np.zeros((self.im_shape[0], self.im_shape[1], 3), dtype=np.float64)
        im_cur = np.zeros((self.im_shape[0], self.im_shape[1], 3), dtype=np.float64)
        traj_grid = np.zeros((self.im_shape[0], self.im_shape[1], 4 if duplicated_traj_code else 2), dtype=np.float64)
        # traj_grid[...]=
        # print(unique_pts)
        # print(mapped_pts)
        # print(traj)
        for pt in unique_pts:
            im_cur[pt[0], pt[1]] = [1, 0, 0]
        # im_cur[unique_pts[0][0], unique_pts[0][1]] = [0, 1, 1]

        # im_prev[mapped_pts[0][0], mapped_pts[0][1]] = [1, 1, 0]
        for pt in mapped_pts:
            im_prev[pt[0], pt[1]] = [1, 0, 0]
        center = torch.tensor(mapped_pts[0][::-1])
        center = center - self.im_shape / 2
        # print(center)
        target_marking = gauss_2d(self.im_shape + ((self.im_shape + 1) % 2), float(0.05 * self.im_shape.prod().sqrt().item()),
                         center.view(-1, 2), (0, 0), density=True)
        target_marking = target_marking/target_marking.max()
        im_prev[:,:,1]=target_marking[0, :self.im_shape[0], :self.im_shape[1]].numpy()
        # print(torch.tensor(unique_pts[0]).view(-1,2), np.sqrt(self.im_shape[0]*self.im_shape[1]))
        center = torch.tensor(unique_pts[0][::-1])
        center = center - self.im_shape / 2
        # print(center)
        label = gauss_2d(self.im_shape + ((self.im_shape + 1) % 2), float(0.05 * self.im_shape.prod().sqrt().item()),
                         center.view(-1, 2), (0, 0), density=True)
        # print(label[0].shape, label.min(), label.max(), label.std(), label.mean())
        # return
        for pt, t in zip(unique_pts, traj):
            traj_grid[pt[0], pt[1]] = t
        totensor = lambda im: torch.tensor(im.transpose((2, 0, 1)), dtype=torch.float32)
        tdict =  TensorDict({'current_image': totensor(im_cur), 'previous_image': totensor(im_prev),
                           'label': torch.tensor(torch.from_numpy(label[0, :self.im_shape[0], :self.im_shape[1]].numpy()), dtype=torch.float32),
                           'trajectory': totensor(traj_grid)/self.im_shape[:,None,None]})
        for k in tdict.keys():
            tdict[k] = tdict[k].unsqueeze(0)
        # print('label',tdict['label'].shape)
        return tdict

#
# import matplotlib.pyplot as plt
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
#
# if __name__ == '__main__':
#     s = TrajectoryPOCSampler((18, 18), 2, traj_disp=5)
#     while True:
#         x = s[0]
#
#         for k in x.keys():
#             print(k, x[k].min(), x[k].max(), x[k].shape)
#         for k in x.keys():
#             x[k]=x[k].squeeze(0)
#             if x[k].dim()==3:
#                 x[k]=x[k].permute(1,2,0)
#             x[k]=x[k].numpy()
#         plt.figure()
#         plt.imshow(np.concatenate([x['current_image']+x['previous_image']+1*np.concatenate([x['label'][..., None]]*3, axis=2)], axis=0))
#         plt.show()
#         plt.imshow(x['trajectory'][:,:,0])
#         plt.show()
#         plt.imshow(x['trajectory'][:,:,1])
#         plt.show()


class SequentialTargetCandidateMatchingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, samples_per_epoch, sup_modes, p_sup_modes=None, processing=no_processing,
                 subseq_modes=None, p_subseq_modes=None, frame_modes=None, p_frame_modes=None):
        """
        args:
            datasets - List of datasets to be used for training
            samples_per_epoch - Number of training samples per epoch
            sup_modes - List of different supervision modes to use (partial_sup or self_sup).
            p_sup_modes - List of sup_mode sample probabilities.
            processing - An instance of Processing class which performs the necessary processing of the data.
            subseq_modes - List of different subsequence modes to sample from (HH, HK, HG), see KeepTrack paper for details.
            p_subseq_modes - List of subseq_mode sample probabilities.
            frame_modes - List of different frame mode to sample from (H, K, J), see KeepTrack paper for details.
            p_frame_modes - List of frame_mode sample probabilities.

        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.processing = processing
        self.subseq_modes = subseq_modes
        self.frame_modes = frame_modes
        self.sup_modes = sup_modes if sup_modes is not None else ['self_sup']
        self.p_sup_modes = p_sup_modes

        if p_sup_modes is None:
            self.p_sup_modes = [1. / len(self.sup_modes)] * (len(self.sup_modes))

        if subseq_modes is not None:
            self.dataset_subseq_states = self._load_dataset_subsequence_states()

            if p_subseq_modes is None:
                p_subseq_modes = [self.dataset_subseq_states[mode].shape[0] for mode in self.subseq_modes]

            # Normalize
            p_subseq_total = sum(p_subseq_modes)
            self.p_subseq_modes = [x / p_subseq_total for x in p_subseq_modes]

        if frame_modes is not None:
            self.dataset_frame_states = self._load_dataset_frame_states()

            if p_frame_modes is None:
                p_frame_modes = [self.dataset_frame_states[mode].shape[0] for mode in self.frame_modes]

            # Normalize
            p_frames_total = sum(p_frame_modes)
            self.p_frame_modes = [x / p_frames_total for x in p_frame_modes]

    def __len__(self):
        return self.samples_per_epoch

    def _load_dataset_subsequence_states(self):
        return self.dataset.get_subseq_states()

    def _load_dataset_frame_states(self):
        return self.dataset.get_frame_states()

    def _sample_valid_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which dumped data is useful

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be sampled
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 2:
            min_id = 2
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        num_begin = num_ids // 2
        num_end = num_ids - num_ids // 2
        ids_begin = random.sample(valid_ids[:len(valid_ids) // 2], k=num_begin)
        ids_end = random.sample(valid_ids[len(valid_ids) // 2:], k=num_end)
        return ids_begin + ids_end

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly).

        returns:
            TensorDict - dict containing all the data blocks
        """

        # select a subseq mode
        sup_mode = random.choices(self.sup_modes, self.p_sup_modes, k=1)[0]

        if sup_mode == 'self_sup':
            mode = random.choices(self.frame_modes, self.p_frame_modes, k=1)[0]

            states = self.dataset_frame_states[mode]
            state = random.choices(states, k=1)[0]
            seq_id = state[0].item()
            baseframe_id = state[1].item()
            test_frame_ids = [baseframe_id]

        elif sup_mode == 'partial_sup':
            mode = random.choices(self.subseq_modes, self.p_subseq_modes, k=1)[0]

            states = self.dataset_subseq_states[mode]
            state = random.choices(states, k=1)[0]
            seq_id = state[0].item()
            baseframe_id = state[1].item()
            test_frame_ids = [baseframe_id, baseframe_id + 1]
        else:
            raise ValueError('Supervision mode: \'{}\' is invalid.'.format(sup_mode))

        seq_info_dict = self.dataset.get_sequence_info(seq_id)

        frames_dict, _ = self.dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({
            'dataset': self.dataset.get_name(),
            'mode': mode,
            'seq_name': self.dataset.get_sequence_name(seq_id),
            'base_frame_id': baseframe_id,
            'sup_mode': sup_mode
        })

        for key, val in frames_dict.items():
            data[key] = val

        return self.processing(data)
