#!/usr/bin/env python3
import dataclasses
from typing import Any, Callable, Iterable
import sys

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


class Drawer:
    def __init__(
        self,
        max_position: int | None = None,
        width: int = 1024,
        height: int | None = None,
        vertical_stretch: float = 1,
        margin: int | None = None,
        spike_stroke_width: float = 4,
        segment_stroke_width: float | None = None,
        arrow_stroke_width: float | None = None,
        arrow_angle: float = np.pi / 6,
        level_step: float = 0.5,
        accelerate_intact: float | None = None,
        font: skia.Font | None = None,
        font_size: int | None = None,
        text_margin: int = 2,
        logarithmic: bool = False,
        arrow_head_size: float = 8,
        background_color=skia.ColorWHITE,
        intact_color=skia.Color(241, 194, 50),
        left_color=skia.Color(140, 173, 237),
        right_color=skia.Color(223, 129, 184),
        level_ticks: bool = False,
    ) -> None:
        self.max_position = max_position
        self.logarithmic = logarithmic  # Early for `self._as_level`.
        self.max_level = self._as_level(self._round_up_to_pow2(self.max_position))
        self.width = width
        self.vertical_stretch = vertical_stretch
        self.height = height or int(
            self.width
            * (1 + (self.max_level - 1) * self.vertical_stretch)
            // self.max_position
        )
        self.vertical_stretch = vertical_stretch
        self.spike_stroke_width = spike_stroke_width
        self.segment_stroke_width = segment_stroke_width or 2 * self.spike_stroke_width
        self.arrow_stroke_width = arrow_stroke_width or self.spike_stroke_width / 2
        self.arrow_angle = arrow_angle

        self.font = skia.Font(skia.Typeface("Arial"), 12) if font is None else font
        if font_size is not None:
            self.font.setSize(font_size)

        self.margin = (
            max(spike_stroke_width, self.font.measureText(str(self.max_position)) / 2)
            if margin is None
            else margin
        )

        self.level_step = level_step
        self.accelerate_intact = (
            (1 if self.logarithmic else 2)
            if accelerate_intact is None
            else accelerate_intact
        )

        self.text_margin = text_margin
        self.text_height = self.font.getSpacing()

        self.arrow_head_size = arrow_head_size

        self.background_color = background_color
        self.intact_color = intact_color
        self.left_color = left_color
        self.right_color = right_color

        self.level_ticks = level_ticks
        self.level_ticks_margin = 0
        if self.level_ticks:
            self.level_ticks_margin = 2 * self.text_margin + self.font.measureText(
                str(self.max_level - 1)
            ) + self.spike_stroke_width

        self.actual_width = self.width - 2 * self.margin - self.level_ticks_margin
        self.actual_height = (
            self.height - 2 * self.margin - self.text_height - self.text_margin
        )

        self.spike_paint = skia.Paint(
            AntiAlias=True,
            Color=skia.ColorBLACK,
            StrokeWidth=self.spike_stroke_width,
            StrokeCap=skia.Paint.kRound_Cap,
        )
        self.arrow_paint = skia.Paint(
            AntiAlias=True,
            Color=skia.ColorGRAY,
            StrokeWidth=self.arrow_stroke_width,
            StrokeCap=skia.Paint.kRound_Cap,
        )
        self.arrow_head_paint = skia.Paint(
            AntiAlias=True, Color=skia.ColorGRAY, Style=skia.Paint.kFill_Style
        )
        self.text_paint = skia.Paint(AntiAlias=True)

    def _lsb(self, x: int) -> int:
        # NB: Returns 0 for `x == 0` while the right answer should be infinity.
        return x & -x

    def _as_level(self, x: int) -> int:
        return x.bit_length() if self.logarithmic else x

    def _into_parts(self, left, right):
        lsb = self._lsb

        right_parts = []
        while right > 0 and right - lsb(right) >= left:
            right_parts.append(
                Part(
                    start=right - lsb(right),
                    end=right,
                    height=lsb(right),
                    is_left=True,
                )
            )
            right -= lsb(right)

        left_parts = []
        while left > 0 and left + lsb(left) <= right:
            left_parts.append(
                Part(start=left, end=left + lsb(left), height=lsb(left), is_left=False)
            )
            left += lsb(left)

        assert left == right
        first_cut_height = lsb(left)
        if first_cut_height == 0:
            first_cut_height = np.inf

        parts = left_parts + right_parts[::-1]
        return Parts(parts=parts, first_cut_height=first_cut_height)

    def _round_up_to_pow2(self, x: int) -> int:
        if x & (x - 1) == 0:  # 0 or a power of 2.
            return x
        return 1 << x.bit_length()

    def _draw_arrowhead(
        self, canvas: skia.Canvas, source: skia.Point, target: skia.Point
    ) -> None:
        line_angle = np.atan2(target.y() - source.y(), target.x() - source.x())

        path = skia.Path()
        path.moveTo(target)
        head_x1 = target.x() - self.arrow_head_size * np.cos(
            line_angle - self.arrow_angle
        )
        head_y1 = target.y() - self.arrow_head_size * np.sin(
            line_angle - self.arrow_angle
        )
        path.lineTo(head_x1, head_y1)
        head_x2 = target.x() - self.arrow_head_size * np.cos(
            line_angle + self.arrow_angle
        )
        head_y2 = target.y() - self.arrow_head_size * np.sin(
            line_angle + self.arrow_angle
        )
        path.lineTo(head_x2, head_y2)
        path.close()

        canvas.drawPath(path, self.arrow_head_paint)

    def _draw_spikes(self, canvas: skia.Canvas) -> None:
        for position in range(self.max_position + 1):
            if position == 0:
                target = self._get_point(position, 0)
                target.fY = 0
            else:
                level = self._as_level(self._lsb(position))
                target = self._get_point(position, level)
            source = self._get_point(position, 0)
            canvas.drawLine(source, target, self.spike_paint)

            text = str(position)
            text_x = source.x() - self.font.measureText(text) / 2
            text_y = (
                self.margin + self.actual_height + self.text_margin + self.text_height
            )
            canvas.drawString(text, text_x, text_y, self.font, self.text_paint)

        if self.level_ticks:
            for level in range(1, self.max_level + 1):
                str_level = str(level - 1)
                delta = self.font.measureText(str_level)
                x = self.margin + self.level_ticks_margin / 2 - delta
                y = self._get_point(0, level).fY + self.text_height / 2
                canvas.drawString(str_level, x, y, self.font, self.text_paint)

    def _get_point(self, position: int, level: float) -> skia.Point:
        x = (
            self.margin
            + self.level_ticks_margin
            + self.actual_width * position / self.max_position
        )
        level_numerator = 0 if level == 0 else 1 + (level - 1) * self.vertical_stretch
        level_denominator = 1 + (self.max_level - 1) * self.vertical_stretch
        y = self.margin + self.actual_height * (1 - level_numerator / level_denominator)
        return skia.Point(x, y)

    def _draw_segment(
        self,
        canvas: skia.Canvas,
        left: int,
        right: int,
        level: float,
        color: skia.Color,
    ) -> None:
        source = self._get_point(left, level)
        target = self._get_point(right, level)
        segment_paint = skia.Paint(
            AntiAlias=True, Color=color, StrokeWidth=self.segment_stroke_width
        )
        canvas.drawLine(source, target, segment_paint)

    def _get_new_surface(self):
        return skia.Surface(self.width, self.height)

    def _as_image(self, surface: skia.Surface) -> skia.Image:
        return surface.makeImageSnapshot()

    def _get_color(self, is_left: bool) -> skia.Color:
        return self.left_color if is_left else self.right_color

    def draw_left_right(self, position: int) -> skia.Image:
        surface = self._get_new_surface()
        with surface as canvas:
            canvas.drawPaint(skia.Paint(Color=self.background_color))
            lsb = self._lsb(position)
            left = position - lsb
            right = position + lsb
            level = self._as_level(lsb)
            self._draw_segment(canvas, left, position, level, self.left_color)
            self._draw_segment(canvas, position, right, level, self.right_color)
            self._draw_spikes(canvas)
        return self._as_image(surface)

    def draw_fenwick_update(self, position: int) -> skia.Image:
        surface = self._get_new_surface()
        with surface as canvas:
            canvas.drawPaint(skia.Paint(Color=self.background_color))

            if position % 2 != 0:
                self._draw_segment(canvas, position, position + 1, 1, self.right_color)

            position += 1
            while position <= self.max_position:
                lsb = self._lsb(position)
                left = position - lsb
                level = self._as_level(lsb)
                self._draw_segment(canvas, left, position, level, self.left_color)
                position += lsb
            self._draw_spikes(canvas)
        return self._as_image(surface)

    def draw_segment_tree(
        self, arrows: bool = False, fenwick: bool = False
    ) -> skia.Image:
        if fenwick and arrows:
            raise ValueError("Arrows are only supported in full segment trees")

        surface = self._get_new_surface()
        with surface as canvas:
            canvas.drawPaint(skia.Paint(Color=self.background_color))

            length = 1
            while length <= self.max_position:
                parent_length = length * 2
                level = self._as_level(length)
                parent_level = self._as_level(parent_length)
                index = 0
                while True:
                    left = index * length
                    right = left + length
                    if right > self.max_position:
                        break
                    is_left = index % 2 == 0
                    if is_left or not fenwick:
                        color = self._get_color(is_left)
                        self._draw_segment(canvas, left, right, level, color)
                    if arrows and level < self.max_level:
                        midpoint = self._get_point((left + right) / 2, level)
                        parent_midpoint = self._get_point(
                            right if is_left else left, parent_level
                        )
                        midpoint.fY -= self.segment_stroke_width / 2
                        parent_midpoint.fY += self.segment_stroke_width / 2
                        canvas.drawLine(parent_midpoint, midpoint, self.arrow_paint)
                        self._draw_arrowhead(canvas, parent_midpoint, midpoint)
                    index += 1
                length = parent_length

            self._draw_spikes(canvas)
        return self._as_image(surface)

    def animate(self, left: int, right: int, start_level: float | None = None) -> Iterable[skia.Image]:
        rec = skia.PictureRecorder()
        canvas = rec.beginRecording(self.width, self.height)
        self._draw_spikes(canvas)
        spikes = rec.finishRecordingAsPicture()

        split = self._into_parts(left, right)

        level = self.max_level if start_level is None else start_level
        while level >= 0:
            surface = self._get_new_surface()
            with surface as canvas:
                canvas.drawPaint(skia.Paint(Color=self.background_color))

                first_cut_level = (
                    np.inf
                    if split.first_cut_height == np.inf
                    else self._as_level(split.first_cut_height)
                )
                intact = level > first_cut_level
                if intact:
                    self._draw_segment(canvas, left, right, level, self.intact_color)
                else:
                    for part in split.parts:
                        actual_level = max(level, self._as_level(part.height))
                        color = self._get_color(part.is_left)
                        self._draw_segment(
                            canvas, part.start, part.end, actual_level, color
                        )

                canvas.drawPicture(spikes)

            yield self._as_image(surface)
            level -= self.level_step * (self.accelerate_intact if intact else 1)


INFINITE_LOOP = 0


def save_animation(
    filename: str,
    frame_generator: Callable[[], Iterable[skia.Image]],
    fps: float = 25,
    loop: int = 1,
    pause_seconds: float = 0.0,
) -> None:
    sys.stderr.write(f"Generating animation {filename}")
    sys.stderr.flush()
    frames = []
    durations = []
    for frame in frame_generator():
        if frame is None:
            durations[-1] += int(1000 * pause_seconds)
        else:
            frames.append(
                frame.convert(
                    alphaType=skia.kUnpremul_AlphaType,
                    colorType=skia.kRGBA_8888_ColorType,
                )
            )
            durations.append(1000 // fps)
        print(".", file=sys.stderr, end="", flush=True)
    print(file=sys.stderr)

    print(f"Storing animation {filename}...", file=sys.stderr)
    imageio.mimwrite(filename, frames, duration=durations, loop=loop)


def add_pause(items: Iterable[Any]) -> Iterable[Any]:
    yield from items
    yield None


def generate_segment_tree_frames(
    drawer: Drawer, pause_frames: int = 0
) -> Iterable[np.ndarray]:
    yield from add_pause(drawer.animate(9, 29))
    yield from add_pause(drawer.animate(3, 37))
    yield from add_pause(drawer.animate(32, 49))
    yield from add_pause(drawer.animate(37, 64))
    yield from add_pause(drawer.animate(0, 15))
    yield from add_pause(drawer.animate(0, 27))


def generate_fenwick_tree_frames(drawer: Drawer, pause_frames: int = 0):
    yield from add_pause(drawer.animate(0, 29))
    yield from add_pause(drawer.animate(0, 37))
    yield from add_pause(drawer.animate(0, 49))
    yield from add_pause(drawer.animate(0, 64))
    yield from add_pause(drawer.animate(0, 15))
    yield from add_pause(drawer.animate(0, 27))


if __name__ == "__main__":
    max_position = 64
    fps = 25
    pause_seconds = 1

    drawer = Drawer(max_position=max_position)
    log_drawer = Drawer(
        max_position=max_position,
        logarithmic=True,
        vertical_stretch=1.5,
        level_step=0.25,
    )
    log_drawer_75 = Drawer(
        max_position=75,
        logarithmic=True,
        vertical_stretch=1.5,
        level_step=0.25,
        font_size=10,
    )
    log_drawer_tall = Drawer(
        max_position=max_position,
        logarithmic=True,
        vertical_stretch=3,
        level_step=0.25,
        level_ticks=True,
    )

    log_drawer_tall.draw_segment_tree(arrows=True).save("segment-tree-arrows.png")
    log_drawer.draw_segment_tree(fenwick=True).save("fenwick-tree.png")
    log_drawer_75.draw_segment_tree(fenwick=True).save("fenwick-tree-75.png")
    log_drawer.draw_fenwick_update(11).save("fenwick-update-11.png")
    drawer.draw_left_right(12).save("left-right-12.png")
    next(log_drawer.animate(3, 37, start_level=0)).save('split-3-37.png')

    save_animation(
        "fenwick-tree-cutting.gif",
        lambda: generate_fenwick_tree_frames(log_drawer),
        fps=fps,
        loop=INFINITE_LOOP,
        pause_seconds=pause_seconds,
    )
    save_animation(
        "segment-tree-cutting.gif",
        lambda: generate_segment_tree_frames(log_drawer),
        fps=fps,
        loop=INFINITE_LOOP,
        pause_seconds=pause_seconds,
    )
    save_animation(
        "fenwick-tree-cutting-expanded.gif",
        lambda: generate_segment_tree_frames(drawer),
        fps=fps,
        loop=INFINITE_LOOP,
        pause_seconds=pause_seconds,
    )
