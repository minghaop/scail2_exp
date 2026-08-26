"""Pure helpers for planning lossless SCAIL-2 temporal segments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSegment:
    """One inference segment and its real/padded temporal boundaries."""

    start: int
    valid_end: int
    padded_frames: int
    overlap: int

    @property
    def valid_frames(self) -> int:
        return self.valid_end - self.start

    @property
    def new_frames(self) -> int:
        return self.valid_frames - self.overlap


def round_up_temporal_frames(frame_count: int, temporal_stride: int = 4) -> int:
    """Round a positive pixel-frame count up to the nearest ``stride*n + 1``."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if temporal_stride <= 0:
        raise ValueError("temporal_stride must be positive")
    return (
        (frame_count - 1 + temporal_stride - 1) // temporal_stride
    ) * temporal_stride + 1


def plan_frame_segments(
    total_frames: int,
    segment_len: int = 81,
    segment_overlap: int = 5,
    temporal_stride: int = 4,
) -> list[FrameSegment]:
    """Plan segments that cover every input frame exactly once after stitching.

    Full segments use the configured overlap. Both segment and history lengths
    must be ``stride*n+1`` so the temporal VAE consumes every supplied frame.
    The final short segment is padded by cloning its last condition frame only
    until its length is valid; padding is removed from the generated result.
    """

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")
    if (segment_len - 1) % temporal_stride:
        raise ValueError(
            f"segment_len must equal {temporal_stride}*n+1, got {segment_len}"
        )
    if not 0 < segment_overlap < segment_len:
        raise ValueError("segment_overlap must be in (0, segment_len)")
    if (segment_overlap - 1) % temporal_stride:
        raise ValueError(
            f"segment_overlap must equal {temporal_stride}*n+1, "
            f"got {segment_overlap}"
        )

    segments: list[FrameSegment] = []
    start = 0
    while start < total_frames:
        valid_end = min(start + segment_len, total_frames)
        valid_frames = valid_end - start
        padded_frames = (
            segment_len
            if valid_frames == segment_len
            else round_up_temporal_frames(valid_frames, temporal_stride)
        )
        overlap = 0 if not segments else segment_overlap
        segment = FrameSegment(start, valid_end, padded_frames, overlap)
        if segment.new_frames <= 0:
            raise RuntimeError(f"Segment adds no new frames: {segment}")
        segments.append(segment)
        if valid_end == total_frames:
            break
        start = valid_end - segment_overlap

    if sum(segment.new_frames for segment in segments) != total_frames:
        raise RuntimeError("Segment plan does not preserve the input frame count")
    return segments
