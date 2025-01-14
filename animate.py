#!/usr/bin/env python3
import dataclasses
from typing import Iterable

import imageio
import numpy as np
import skia

@dataclasses.dataclass
class Part:
    start: int
    end: int
    height: int
    is_left: bool

@dataclasses.dataclass
class Parts:
    parts: list[Part]
    first_cut_height: float

def lsb(x: int) -> int:
    # Note: Returns 0 for `x == 0` while the right answer should be infinity.
    return x & -x

def into_parts(left, right):
    def _prev(x):
        return x - lsb(x)
    def _next(x):
        return x + lsb(x)

    right_parts = []
    while right > 0 and _prev(right) >= left:
        right_parts.append(Part(start=_prev(right), end=right,
                                height=lsb(right), is_left=False))
        right = _prev(right)

    left_parts = []
    while left > 0 and _next(left) <= right:
        left_parts.append(Part(start=left, end=_next(left),
                               height=lsb(left), is_left=True))
        left = _next(left)

    assert left == right
    first_cut_height = lsb(left)
    if first_cut_height == 0:
        first_cut_height = np.inf

    parts = left_parts + right_parts[::-1]
    return Parts(parts=parts, first_cut_height=first_cut_height)

def round_up_to_pow2(x: int) -> int:
    return 1 << x.bit_length()

def animate(
    left: int,
    right: int,
    max_position: int | None = None,
    width: int = 1024,
    height: int | None = None,
    margin: int | None = None,
    slicer_stroke_width: float = 4,
    segment_stroke_width: float | None = None,
    level_step: float = 0.5,
    accelerate_intact: float | None = None,
    font: skia.Font | None = None,
    text_margin: int = 2,
    logarithmic: bool = False
) -> Iterable[np.ndarray]:
    def as_level(x: int) -> int:
        return x.bit_length() if logarithmic else x

    height = height or width * as_level(max_position) // max_position
    if max_position is None:
        max_position = round_up_to_pow2(right)
    segment_stroke_width = segment_stroke_width or 2 * slicer_stroke_width
    if font is None:
        font = skia.Font(skia.Typeface('Arial'), 12)
    if margin is None:
        margin = max(slicer_stroke_width,
                     font.measureText(str(max_position)) / 2)
    if accelerate_intact is None:
        accelerate_intact = 1 if logarithmic else 2

    text_height = font.getSpacing()

    actual_width = width - 2 * margin
    actual_height = height - 2 * margin - text_height - text_margin

    slicer_paint = skia.Paint(
            AntiAlias=True, Color=skia.ColorBLACK,
            StrokeWidth=slicer_stroke_width, StrokeCap=skia.Paint.kRound_Cap)
    text_paint = skia.Paint(AntiAlias=True)
    rec = skia.PictureRecorder()
    canvas = rec.beginRecording(width, height)

    max_level = as_level(max_position)
    for position in range(max_position + 1):
        if position == 0:
            value = max_position
        else:
            value = lsb(position)
        x0 = margin + actual_width * position / max_position
        y0 = margin + actual_height
        x1 = x0
        y1 = margin + actual_height * (1 - as_level(value) / max_level)
        canvas.drawLine(skia.Point(x0, y0), skia.Point(x1, y1), slicer_paint)

        text = str(position)
        text_x = x0 - font.measureText(text) / 2
        text_y = margin + actual_height + text_margin + text_height
        canvas.drawString(text, text_x, text_y, font, text_paint)
    slicers = rec.finishRecordingAsPicture()

    split = into_parts(left, right)

    segment_paint = skia.Paint(
            AntiAlias=True, Color=skia.ColorMAGENTA,
            StrokeWidth=segment_stroke_width)
    level = as_level(max_position)
    while level > 0:
        surface = skia.Surface(width, height)
        with surface as canvas:
            canvas.drawPaint(skia.Paint(Color=skia.ColorWHITE))

            first_cut_level = (np.inf
                               if split.first_cut_height == np.inf
                               else as_level(split.first_cut_height))
            intact = level > first_cut_level
            if intact:
                x0 = margin + actual_width * left / max_position
                y0 = margin + actual_height * (1 - level / max_level)
                x1 = margin + actual_width * right / max_position
                y1 = y0
                canvas.drawLine(
                        skia.Point(x0, y0), skia.Point(x1, y1), segment_paint)
            else:
                for part in split.parts:
                    actual_level = max(level, as_level(part.height))
                    x0 = margin + actual_width * part.start / max_position
                    y0 = margin + actual_height * (1 - actual_level / max_level)
                    x1 = margin + actual_width * part.end / max_position
                    y1 = y0
                    segment_paint.setColor(skia.ColorBLUE if part.is_left
                                           else skia.ColorRED)
                    canvas.drawLine(skia.Point(x0, y0), skia.Point(x1, y1),
                                    segment_paint)

            canvas.drawPicture(slicers)

        yield surface.makeImageSnapshot().toarray()
        level -= level_step * (accelerate_intact if intact else 1)

def with_pause_at_end(
    frames: Iterable[np.ndarray], pause_frames: int
) -> Iterable[np.ndarray]:
    for frame in frames:
        yield frame
    last_frame = frame
    for _ in range(pause_frames):
        yield last_frame

def save_animation(
    filename: str,
    frames: Iterable[np.ndarray],
    fps: float = 25,
    loop: int = 1
) -> None:
    with imageio.get_writer(
            filename, mode='I', fps=fps, loop=loop) as writer:
        for frame in frames:
            writer.append_data(frame)

def generate_segment_tree_frames(
        max_position: int, pause_frames: int = 0, logarithmic: bool = False):
    def _generate(left: int, right: int) -> Iterable[np.ndarray]:
        yield from with_pause_at_end(
                animate(left, right, max_position=max_position,
                        logarithmic=logarithmic), pause_frames)
    yield from _generate(9, 29)
    yield from _generate(3, 37)
    yield from _generate(32, 49)
    yield from _generate(37, 64)
    yield from _generate(0, 15)
    yield from _generate(0, 27)

def generate_fenwick_tree_frames(max_position: int, pause_frames: int = 0):
    def _generate(right: int) -> Iterable[np.ndarray]:
        yield from with_pause_at_end(
                animate(0, right, max_position=max_position, logarithmic=True),
                pause_frames)
    yield from _generate(29)
    yield from _generate(37)
    yield from _generate(49)
    yield from _generate(64)
    yield from _generate(15)
    yield from _generate(27)

if __name__ == '__main__':
    save_animation('segment-tree-spikes.gif',
                   generate_segment_tree_frames(max_position=64,
                                                pause_frames=25), loop=0)
    save_animation('fenwick-tree-spikes.gif',
                   generate_fenwick_tree_frames(max_position=64,
                                                pause_frames=25), loop=0)
    save_animation('segment-tree-spikes-log.gif',
                   generate_segment_tree_frames(max_position=64,
                                                pause_frames=25,
                                                logarithmic=True), loop=0)
