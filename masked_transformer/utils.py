import torch


def temporal_iou(target_segments, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 2d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    if len(target_segments.shape) == 1:
        target_segments = target_segments.unsqueeze(0)
    if len(candidate_segments.shape) == 1:
        candidate_segments = candidate_segments.unsqueeze(0)
    tt1 = torch.max(target_segments[:, 0:1], candidate_segments[:, 0])
    tt2 = torch.min(target_segments[:, 1:2], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clamp(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
        + (target_segments[:, 1:2] - target_segments[:, 0:1]) \
        - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    iou = segments_intersection.float() / segments_union
    return iou


def temporal_nms(segments, scores, thresh=0.3, n_prop=None,
                 return_indices=False):
    if n_prop is None:
        n_prop = segments.shape[0]
    indices = torch.argsort(scores, descending=True)
    segments = segments[indices]

    nms_segments = segments.new(segments.shape)
    nms_indices = segments.new(indices.shape)
    cnt = 0
    for seg_ind, segment in zip(indices, segments):
        if cnt == 0 or torch.max(temporal_iou(segment, nms_segments[:cnt])) < thresh:  # noqa: E501
            nms_segments[cnt] = segment
            nms_indices[cnt] = seg_ind
            cnt += 1
            if cnt >= n_prop:
                break

    if return_indices:
        return nms_segments[:cnt], nms_indices[:cnt].long()
    return nms_segments[:cnt]
