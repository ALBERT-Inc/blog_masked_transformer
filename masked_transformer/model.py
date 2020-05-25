import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from masked_transformer.utils import temporal_iou, temporal_nms
from masked_transformer.serializer import load_pth_renamed


class TemporalCoder:
    def __init__(self, kernel_list, slide_window_size=480, stride_factor=50,
                 pos_thresh=0.7, neg_thresh=0.1):
        self.kernel_list = kernel_list
        self.slide_window_size = slide_window_size
        self.stride_factor = stride_factor
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self._init_anchor()

    def encode(self, segments, all_segments=None):
        """
        Args:
            segments: shape is ``(B, 2)``.
            all_segments: list of Tensor of which shape is ``(N, 2)`` where
                ``N`` is the number of segments of each video.
        """
        iou = temporal_iou(segments, self.anchor_segments)
        pos_mask = (iou >= self.pos_thresh).to(torch.float32)
        cen_off = ((segments[:, 1:2] + segments[:, 0:1]) / 2 - self.anc_cen) / self.anc_len  # noqa: E501
        len_off = torch.log((segments[:, 1:2] - segments[:, 0:1]) / self.anc_len)  # noqa: E501

        if all_segments is None:
            return cen_off, len_off, pos_mask
        neg_mask = []
        for all_segment in all_segments:
            all_iou, _ = temporal_iou(all_segment, self.anchor_segments).max(0)
            neg_mask.append((all_iou < self.neg_thresh))
        return cen_off, len_off, pos_mask, torch.stack(neg_mask).float()

    def decode(self, cen_off, len_off):
        prop_cen = self.anc_cen + cen_off * self.anc_len
        prop_len = self.anc_len * torch.exp(len_off)
        prop_start = prop_cen - prop_len / 2
        prop_end = prop_cen + prop_len / 2
        return prop_start, prop_end

    def to(self, device):
        self.anchor_segments = self.anchor_segments.to(device)
        self.anc_len = self.anc_len.to(device)
        self.anc_cen = self.anc_cen.to(device)

    def _init_anchor(self):
        anc_len_lst = []
        anc_cen_lst = []
        for k_size in self.kernel_list:
            anc_cen = torch.arange(k_size / 2,
                                   self.slide_window_size + 1 - k_size / 2,
                                   math.ceil(k_size / self.stride_factor))
            anc_len = torch.full(anc_cen.shape, k_size)
            anc_len_lst.append(anc_len)
            anc_cen_lst.append(anc_cen)
        self.anc_len = torch.cat(anc_len_lst)
        self.anc_cen = torch.cat(anc_cen_lst)
        self.anchor_segments = torch.stack([self.anc_cen - self.anc_len / 2,
                                            self.anc_cen + self.anc_len / 2], 1)  # noqa: #501


class DropoutTime1D(nn.Module):
    '''
    assumes the first dimension is batch,
    input in shape B x T x H
    '''
    def __init__(self, p_drop=0.1):
        super(DropoutTime1D, self).__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if self.training:
            mask = x.data.new(x.data.size(0), x.data.size(1), 1).uniform_()
            mask = (mask > self.p_drop).float()
            return x * mask
        else:
            return x * (1-self.p_drop)

    def init_params(self):
        pass

    def __repr__(self):
        repstr = self.__class__.__name__ + ' (\n'
        repstr += "{:.2f}".format(self.p_drop)
        repstr += ')'
        return repstr


class MaskedTransformer(nn.Module):
    def __init__(self, coder, vocab, d_model, d_hidden, n_layers=2, n_heads=8,
                 n_words=20, pos_thresh=0.7, max_n_prop=500, nms_thresh=0.45,
                 weights_file=None, correspondence_file=None):
        super(MaskedTransformer, self).__init__()
        self.n_words = n_words
        self.pos_thresh = pos_thresh
        self.nms_thresh = nms_thresh
        self.max_n_prop = max_n_prop
        self.coder = coder
        self.vocab = vocab
        self.rgb_emb = nn.Linear(d_hidden, d_model // 2)
        self.flow_emb = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Sequential(
            DropoutTime1D(),
            nn.ReLU()
        )
        self.transformer = Transformer(d_model,
                                       d_hidden=d_hidden,
                                       n_layers=n_layers,
                                       n_heads=n_heads,
                                       drop_ratio=0.1)
        self.proposal_decoder = ProposalDecoder(coder.kernel_list, d_model)
        self.mask_model = MaskModel(coder.anchor_segments, d_model,
                                    coder.slide_window_size)

        self.caption_decoder = CaptionDecoder(self.transformer,
                                              d_model,
                                              vocab,
                                              d_hidden=d_hidden,
                                              n_layers=n_layers,
                                              n_heads=n_heads,
                                              drop_ratio=0.1)
        self.output_type = 'words'
        self.init_weight(weights_file, correspondence_file)

    def forward(self, rgb, flow, sentence, segments=None):
        if segments is not None and self.training:
            _, _, pos_mask = self.coder.encode(segments)
        else:
            pos_mask = None
        y_rgb = self.rgb_emb(rgb)
        y_flow = self.flow_emb(flow)
        vis_feat = self.dropout(torch.cat((y_rgb, y_flow), 2))
        encodings = self.transformer(vis_feat)

        # Proposal Decoder
        prop_scores, off_cens, off_lens = self.proposal_decoder(encodings[-1])
        prop_starts, prop_ends = self.coder.decode(off_cens, off_lens)
        if pos_mask is not None and 0 not in pos_mask.sum(1):
            prop_idx = torch.multinomial(pos_mask, 1)[:, 0]
        else:
            prop_idx = prop_scores.argmax(dim=1)
        prop_start = torch.gather(prop_starts, 1, prop_idx.unsqueeze(-1))
        prop_end = torch.gather(prop_ends, 1, prop_idx.unsqueeze(-1))
        prop_seg = torch.cat([prop_start, prop_end], 1)
        pred_mask, gated_mask = self.mask_model(prop_seg,
                                                prop_scores,
                                                y_rgb.shape[1],
                                                prop_idx)

        # Caption Decoder
        word_prob = self.caption_decoder(sentence, encodings, gated_mask)
        prop_off = torch.stack([off_cens, off_lens], -1)
        return word_prob, prop_scores, prop_seg, prop_off, pred_mask

    def to(self, *args, **kwargs):
        super(MaskedTransformer, self).to(*args, **kwargs)
        device = next(self.parameters()).device
        self.coder.to(device)

    def predict(self, rgb, flow):
        y_rgb = self.rgb_emb(rgb)
        y_flow = self.flow_emb(flow)
        vis_feat = self.dropout(torch.cat((y_rgb, y_flow), 2))
        encodings = self.transformer(vis_feat)

        # Proposal Decoder
        prop_scores, off_cens, off_lens = self.proposal_decoder(encodings[-1])
        prop_scores = torch.sigmoid(prop_scores)
        prop_starts, prop_ends = self.coder.decode(off_cens, off_lens)

        sentences = []
        segments = []
        for i in range(y_rgb.shape[0]):
            pos_props = prop_scores[i] >= self.pos_thresh
            pos_prop_starts = prop_starts[i, pos_props]
            pos_prop_ends = prop_ends[i, pos_props]
            pos_inds = torch.where(pos_props)[0]
            pos_seg = torch.stack([pos_prop_starts, pos_prop_ends], 1)
            final_seg, final_ind = temporal_nms(
                pos_seg, prop_scores[i, pos_props], self.nms_thresh,
                n_prop=self.max_n_prop, return_indices=True
            )
            if final_seg.shape[0] == 0:
                segments.append(final_seg.new_empty((0, 2)))
                sentences.append([])
                continue
            pos_inds = pos_inds[final_ind]
            _, mask = self.mask_model(final_seg, prop_scores[i],
                                      y_rgb.shape[1], pos_inds)

            batch_encoding = vis_feat[i].expand(mask.shape[0],
                                                *vis_feat[i].shape)
            pred_sentence = self.caption_decoder.predict(
                batch_encoding, mask, self.n_words
            )
            pred_sentence = self._denum(pred_sentence)
            segments.append(final_seg)
            sentences.append(pred_sentence)
        return sentences, segments

    def _denum(self, sentences):
        if self.output_type == 'words':
            eos_id = self.vocab.stoi['<eos>']
            return [[self.vocab.itos[i] for i in sen if i != eos_id]
                    for sen in sentences]
        else:
            return [' '.join(self.vocab.itos[i] for i in sen).replace(' <eos>', '')  # noqa: E501
                    for sen in sentences]

    def use_preset(self, mode='evaluate'):
        if mode == 'evaluate':
            self.output_type = 'words'
        elif mode == 'visualize':
            self.output_type = 'sentence'
        else:
            raise ValueError('mode must be visualize or evalaute')

    def init_weight(self, weight_file=None, correspondence_file=None):
        if weight_file:
            if correspondence_file:
                with open(correspondence_file, "r", encoding="utf-8") as f:
                    correspondence = json.load(f)
                load_pth_renamed(self, torch.load(weight_file), correspondence)
            else:
                params = torch.load(weight_file)
                self.load_state_dict(params)


class MaskModel(nn.Module):
    def __init__(self, anchor_segments, d_model, window_length):
        super(MaskModel, self).__init__()
        self.anchor_segments = anchor_segments
        self.d_model = d_model
        self.mask_model = nn.Sequential(
            nn.Linear(d_model + window_length, d_model, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, window_length)
        )

    def forward(self, prop_seg, scores, time, prop_idx):
        anchor_segments = self.anchor_segments[prop_idx]
        if prop_seg.is_cuda:
            anchor_segments = anchor_segments.cuda(prop_seg.get_device())
        bin_anchor = bin_time(anchor_segments, time)

        in_pred_mask = torch.cat([
            positional_encodings(prop_seg[:, 0], self.d_model // 4),
            positional_encodings(prop_seg[:, 1], self.d_model // 4),
            positional_encodings(anchor_segments[:, 0], self.d_model // 4),
            positional_encodings(anchor_segments[:, 1], self.d_model // 4),
            bin_anchor], 1
        )
        pred_mask = torch.sigmoid(self.mask_model(in_pred_mask))
        bin_prop = bin_time(prop_seg, time)
        if self.training:
            scores = torch.gather(scores, 1, prop_idx.unsqueeze(-1))
        else:
            scores = scores[prop_idx].unsqueeze(-1)
        gated_mask = scores * bin_prop + (1 - scores) * pred_mask
        return pred_mask, gated_mask


class CaptionDecoder(nn.Module):
    def __init__(self, encoder, d_model, vocab, d_hidden, n_layers, n_heads,
                 drop_ratio=0.1):
        super(CaptionDecoder, self).__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)
        self.out = nn.Linear(d_model, len(vocab))
        self.vocab = vocab
        self.d_model = d_model

    def forward(self, sentence, encodings, mask):
        x = F.embedding(sentence[:, :-1], self.out.weight * math.sqrt(self.d_model))  # noqa: E501
        positions = torch.arange(0, x.shape[1]).float()
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
        x = x + positional_encodings(positions, x.shape[-1])
        h = self.dropout(x)
        for layer, encoding in zip(self.layers, encodings):
            encoding = mask.unsqueeze(-1) * encoding
            h = layer(h, encoding)
        h = remove_pad(self.vocab, sentence[:, 1:], h)
        logits = self.out(h)
        return logits

    def predict(self, x, mask, n_words):

        encodings = self.encoder(x, mask.unsqueeze(-1))

        B, _, H = encodings[0].size()
        embeded = self.out.weight * math.sqrt(self.d_model)
        prediction = encodings[0].new(B, n_words).long().fill_(self.vocab.stoi['<pad>'])  # noqa: E501
        hiddens = [encodings[0].new(B, n_words, H).zero_()
                   for _ in range(len(self.layers) + 1)]
        positions = torch.arange(0, n_words).float()
        if embeded.is_cuda:
            positions = positions.cuda(embeded.get_device())
        hiddens[0] = hiddens[0] + positional_encodings(positions, H)
        for w in range(n_words):
            if w == 0:
                init = embeded.new_full((B, ), self.vocab.stoi['<init>']).long()  # noqa
                hiddens[0][:, w] = hiddens[0][:, w] + \
                    F.embedding(init, embeded)
            else:
                hiddens[0][:, w] = hiddens[0][:, w] + \
                    F.embedding(prediction[:, w - 1], embeded)

            hiddens[0][:, w] = self.dropout(hiddens[0][:, w])
            for layer in range(len(self.layers)):
                encoding = encodings[layer]
                x = hiddens[layer][:, :w + 1]
                x = self.layers[layer].selfattn(hiddens[layer][:, w], x, x)
                hiddens[layer + 1][:, w] = self.layers[layer].feedforward(
                    self.layers[layer].attention(x, encoding, encoding))
            _, prediction[:, w] = self.out(hiddens[-1][:, w]).max(-1)
        return prediction


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads=8, drop_ratio=0.1):
        super(DecoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio, sequential=True),
            d_model, drop_ratio
        )
        self.attention = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio
        )
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x, encoding):
        x = self.selfattn(x, x, x)
        return self.feedforward(self.attention(x, encoding, encoding))


class ProposalDecoder(nn.Module):
    def __init__(self, kernel_list, d_model, stride_factor=50):
        super(ProposalDecoder, self).__init__()
        self.kernel_list = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model,
                          kernel_list[i],
                          stride=math.ceil(kernel_list[i]/stride_factor),
                          groups=d_model,
                          bias=False),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model, 1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, 3, 1)
            )
            for i in range(len(kernel_list))
        ])

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        prop_scores = []
        prop_centers = []
        prop_lens = []
        for kernel in self.kernel_list:
            proposals = kernel(x)
            prop_scores.append(proposals[:, 0])
            prop_centers.append(proposals[:, 1])
            prop_lens.append(proposals[:, 2])
        prop_scores = torch.sigmoid(torch.cat(prop_scores, 1))
        prop_centers = torch.cat(prop_centers, 1)
        prop_lens = torch.cat(prop_lens, 1)
        return prop_scores, prop_centers, prop_lens


class Transformer(nn.Module):

    def __init__(self, d_model, d_hidden, n_layers=2, n_heads=8,
                 drop_ratio=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, mask=None):
        positions = torch.arange(0, x.shape[1]).float()
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
        x = x + positional_encodings(positions, x.shape[-1])
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        encoding = []
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask
            encoding.append(x)
        return encoding


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):
    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        # First element of x is used for skip connection.
        # This might be used for self attention or word embedding.
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))


class Attention(nn.Module):
    def __init__(self, d_key, drop_ratio, sequential=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.sequential = sequential

    def forward(self, query, key, value):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and self.sequential:
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * 1e10
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))
        softmax = F.softmax(dot_products / self.scale, dim=-1)
        return matmul(self.dropout(softmax), value)


class MultiHead(nn.Module):
    def __init__(self, d_key, d_value, n_heads, drop_ratio, sequential=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, sequential)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (x.chunk(self.n_heads, -1)
                             for x in (query, key, value))
        heads = [self.attention(q, k, v) for q, k, v in zip(query, key, value)]
        heads = torch.cat(heads, -1)
        return self.wo(heads)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))


def positional_encodings(positions, n_ch):
    if (len(positions.shape) == 2) and (positions.shape[1] == 1):
        pass
    elif len(positions.shape) == 1:
        positions = positions.unsqueeze(-1)
    else:
        raise ValueError("positions must be 1D vector or 2D of which"
                         "last dimention is 1.")

    encodings = torch.zeros((len(positions), n_ch))
    even_ch = torch.arange(0, n_ch, 2).float()

    if positions.is_cuda:
        encodings = encodings.cuda(positions.get_device())
        even_ch = even_ch.cuda(positions.get_device())

    delim = 10000 ** (even_ch / n_ch)

    encodings[:, ::2] = torch.sin(positions / delim)
    encodings[:, 1::2] = torch.cos(positions / delim)
    return encodings


def bin_time(start_end, time):
    time_index = torch.arange(0, time)
    if start_end.is_cuda:
        time_index = time_index.cuda(start_end.get_device())
    return ((start_end[:, 0].unsqueeze(-1) <= time_index) *
            (start_end[:, 1].unsqueeze(-1) > time_index)).float()


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


def remove_pad(vocab, sentence, out=None):
    pad_idx = vocab.stoi['<pad>']
    mask = sentence != pad_idx
    if out is None:
        return sentence[mask]
    mask = mask.unsqueeze(-1).expand_as(out)
    return out[mask].view(-1, out.size(-1))


class MaskedTransformerLoss(nn.Module):
    def __init__(self, coder, vocab, weight_event=1, weight_mask=1,
                 weight_sentence=1, k=1):
        super(MaskedTransformerLoss, self).__init__()
        self.coder = coder
        self.vocab = vocab
        self.weight_event = weight_event
        self.weight_mask = weight_mask
        self.weight_sentence = weight_sentence
        self.k = k

    def forward(self, preds, targets):
        word_prob, prop_scores, prop_seg, prop_off, pred_mask = preds
        time = pred_mask.shape[-1]

        sentence, segments, all_segments = targets
        sentence = remove_pad(self.vocab, sentence[:, 1:])
        gt_off_cen, gt_off_len, pos_mask, neg_mask = \
            self.coder.encode(segments, all_segments)

        loss_off_cen = F.smooth_l1_loss(prop_off[..., 0], gt_off_cen,
                                        reduction='none')
        loss_off_len = F.smooth_l1_loss(prop_off[..., 1], gt_off_len,
                                        reduction='none')
        loss_off_cen *= pos_mask
        loss_off_len *= pos_mask
        n_pos = pos_mask.sum()
        loss_off_cen = torch.sum(loss_off_cen) / n_pos
        loss_off_len = torch.sum(loss_off_len) / n_pos
        loss_event = self._event_loss(prop_scores, pos_mask, neg_mask)
        loss_mask = F.binary_cross_entropy(pred_mask, bin_time(prop_seg, time))
        loss_sentence = F.cross_entropy(word_prob, sentence)
        return loss_off_cen + loss_off_len + loss_event * self.weight_event + \
            loss_mask * self.weight_mask + loss_sentence * self.weight_sentence

    def _event_loss(self, scores, positive, negative):
        n_positive = positive.sum()
        conf_loss = F.binary_cross_entropy(scores, positive, reduction='none')
        hard_negative = self._hard_negative(conf_loss, negative,
                                            positive.sum(axis=1))
        n_negative = hard_negative.sum()
        # logical or
        conf_loss *= (positive + hard_negative)
        conf_loss = torch.sum(conf_loss) / (n_positive + n_negative)
        return conf_loss

    def _hard_negative(self, x, negative, n_pos):
        """Hard Negative Mining.
        Args:
            x (torch.Tensor): Tensor of shape ``(n, n_b) where
                ``n`` is batchsize and ``n_b`` is the number of multibox.
            positive (torch.BoolTensor): Tensor of shape ``(n, n_b)`` where
                ``n`` is batchsize and ``n_b`` is the number of multibox.
            k (int or float): Parameter for hard negative mining.
        Returns:
            torch.BoolTensor: Flag of hard negative of shape ``(n, n_b)`` where
                ``n`` is batchsize and ``n_b`` is the number of multibox.
        """
        # rank negative loss with lower order
        rank = (-x * negative.long()).argsort(axis=1).argsort(axis=1)
        hard_negative = rank < (n_pos * self.k).unsqueeze(1)
        return hard_negative
