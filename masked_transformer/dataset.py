import json
import math
from pathlib import Path
from collections import defaultdict

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchtext
import numpy as np


class YouCook(Dataset):
    def __init__(self, root, text_proc, image_set='train',
                 slide_window_size=480, transforms=None, **kwargs):
        self.root = Path(root)
        self.annotation = self.root / 'yc2' / 'yc2_annotations_trainval.json'
        self.image_set = self._init_imageset(image_set)
        self.slide_window_size = slide_window_size
        self.text_proc = text_proc
        self.vocab = self.text_proc.vocab
        dur_file = self.root / 'yc2' / 'yc2_duration_frame.csv'
        self.sampling_sec = self._calc_true_sampling_sec(dur_file)
        self.vids, self.sentences, self.segments,\
            self.urls, self.all_segments = self._get_samples(self.sampling_sec)
        self.collate_fn = self.train_collate_fn \
            if self.image_set == 'training' else self.test_collate_fn

    @staticmethod
    def _init_imageset(image_set):
        if image_set == 'train':
            return 'training'
        elif image_set == 'val':
            return 'validation'
        elif image_set == 'test':
            return 'test'
        else:
            raise ValueError("Invalid image_set name")

    def _get_samples(self, sampling_sec):
        with open(self.annotation, 'r') as data_file:
            annos = json.load(data_file)
        sentences = []
        segments = []
        all_segments = defaultdict(list)
        vids = []
        urls = []
        for vid, values in annos['database'].items():
            if values['subset'] == self.image_set:
                if self.image_set == 'training':
                    for ann in values['annotations']:
                        start_sec, end_sec = ann['segment']
                        if end_sec <= self.slide_window_size * sampling_sec[vid]:  # noqa: E501
                            vids.append(vid)
                            sentences.append(ann['sentence'].strip())
                            gt_start = start_sec / sampling_sec[vid]
                            gt_end = end_sec / sampling_sec[vid]
                            segments.append((gt_start, gt_end))
                            urls.append(values['video_url'])
                            all_segments[vid].append((gt_start, gt_end))
                else:
                    vids.append(vid)
                    urls.append(values['video_url'])
                    video_sentence = []
                    video_segment = []
                    for ann in values['annotations']:
                        start_sec, end_sec = ann['segment']
                        video_sentence.append(ann['sentence'].strip())
                        gt_start = start_sec / sampling_sec[vid]
                        gt_end = end_sec / sampling_sec[vid]
                        video_segment.append((gt_start, gt_end))
                    sentences.append(video_sentence)
                    segments.append(video_segment)
                    all_segments[vid].append(video_segment)
        return vids, sentences, segments, urls, all_segments

    def _calc_true_sampling_sec(self, dur_file, sampling_sec=0.5):
        """Calculate true sampling second.
        Youcook samples each video frame features every 0.5s, but it is not
        accurate due to int casting.
        See following issue https://github.com/salesforce/densecap/issues/27.
        """
        true_sampling_sec = {}
        with open(dur_file) as f:
            for line in f:
                vid, dur, frame = [elem.strip() for elem in line.split(',')]
                true_sampling_sec[vid] = float(dur) / float(frame) * \
                    math.ceil(float(frame) / float(dur) * sampling_sec)
        return true_sampling_sec

    def stoi(self, sentence):
        if isinstance(sentence, list):
            sentence = list(map(self.text_proc.preprocess, sentence))
        else:
            sentence = [self.text_proc.preprocess(sentence)]
        return self.text_proc.numericalize(self.text_proc.pad(sentence))

    def __getitem__(self, idx):
        vid_prefix = str(self.root / self.image_set / self.vids[idx])
        if self.image_set == 'training':
            sentence = self.stoi(self.sentences[idx])[0]
        else:
            sentence = [sen.replace('  ', ' ').split(' ')
                        for sen in self.sentences[idx]]
        segment = torch.Tensor(self.segments[idx])

        resnet_feat = torch.from_numpy(np.load(vid_prefix + '_resnet.npy')).float()  # noqa: E501
        bn_feat = torch.from_numpy(np.load(vid_prefix + '_bn.npy')).float()
        if resnet_feat.shape[0] < self.slide_window_size:
            pad_size = self.slide_window_size - resnet_feat.shape[0]
            resnet_feat = F.pad(resnet_feat, [0, 0, 0, pad_size], value=0)
            bn_feat = F.pad(bn_feat, [0, 0, 0, pad_size], value=0)
        else:
            resnet_feat = resnet_feat[:self.slide_window_size]
            bn_feat = bn_feat[:self.slide_window_size]

        all_segment = torch.Tensor(self.all_segments[self.vids[idx]])
        return resnet_feat, bn_feat, sentence, segment, all_segment

    def __len__(self):
        return len(self.vids)

    @staticmethod
    def get_text_proc(root, max_length=20, min_freq=5, stop_words=None):
        # build vocab and tokenized sentences
        text_proc = torchtext.data.Field(sequential=True, init_token='<init>',
                                         eos_token='<eos>', tokenize='spacy',
                                         lower=True, batch_first=True,
                                         fix_length=max_length,
                                         stop_words=stop_words)
        with open(Path(root) / 'yc2' / 'yc2_annotations_trainval.json', 'r') as f:  # noqa: E501
            annotations = json.load(f)

        train_val_sentences = []
        for vid, val in annotations['database'].items():
            if val['subset'] in ['training', 'validation']:
                for ind, ann in enumerate(val['annotations']):
                    train_val_sentences.append(ann['sentence'].strip())

        sentences_proc = list(map(text_proc.preprocess, train_val_sentences))
        text_proc.build_vocab(sentences_proc, min_freq=min_freq)
        return text_proc

    def train_collate_fn(self, batch):
        batched = []
        for i, sample in enumerate(zip(*batch)):
            if i == 4:
                batched.append(sample)
            else:
                batched.append(torch.stack(sample))
        return batched

    def test_collate_fn(self, batch):
        batched = []
        for i, sample in enumerate(zip(*batch)):
            if i < 2:
                batched.append(torch.stack(sample))
            else:
                batched.append(sample)
        return batched
