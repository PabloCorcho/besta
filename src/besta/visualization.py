from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Optional, Union, List
import numpy as np


def draw_dict_in_axes(
    ax,
    info: Union[Mapping, Sequence[Mapping], Sequence[Tuple[str, Mapping]]],
    loc: Tuple[float, float] = (0.02, 0.98),
    max_fontsize: int = 14,
    min_fontsize: int = 6,
    family: str = "monospace",
    ha: str = "left",
    va: str = "top",
    line_spacing: float = 1.0,
    section_spacing: int = 1,
    title_style: str = "plain",  # "plain" | "underline"
    sort_keys: bool = False,
):
    """
    Render one or more dictionaries as readable text inside a matplotlib Axes,
    automatically adjusting font size so the text fits within the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.

    info : dict or sequence
        One of:
          - dict: a single section without a title
          - list/tuple of dict: multiple sections (auto-titled "Section 1", ...)
          - list/tuple of (title, dict): multiple titled sections

        Each dict should map keys to numerical (or printable) values.

    loc : tuple, optional
        (x, y) location in axes coordinates (default top-left).

    max_fontsize : int, optional
        Starting (largest) font size.

    min_fontsize : int, optional
        Minimum font size allowed.

    family : str, optional
        Font family (monospace recommended).

    ha, va : str, optional
        Horizontal / vertical alignment.

    line_spacing : float, optional
        Line spacing multiplier.

    section_spacing : int, optional
        Number of blank lines between sections.

    title_style : {"plain","underline"}, optional
        How to render section titles.

    sort_keys : bool, optional
        If True, sort keys within each section.

    Returns
    -------
    text : matplotlib.text.Text
        The text artist.
    """

    # ------------------------- normalize input -------------------------
    sections: List[Tuple[Optional[str], Mapping]] = []

    if isinstance(info, Mapping):
        sections = [(None, info)]
    else:
        # sequence-like
        info_list = list(info)
        if len(info_list) == 0:
            sections = [(None, {})]
        else:
            # if elements look like (title, dict)
            first = info_list[0]
            if (
                isinstance(first, (tuple, list))
                and len(first) == 2
                and isinstance(first[0], str)
                and isinstance(first[1], Mapping)
            ):
                sections = [(str(t), d) for (t, d) in info_list]
            else:
                # assume it's a list of dicts
                if not all(isinstance(d, Mapping) for d in info_list):
                    raise TypeError(
                        "info must be a dict, a sequence of dicts, or a sequence of (title, dict)."
                    )
                sections = [(f"Section {i+1}", d) for i, d in enumerate(info_list)]

    # ------------------------- format text -------------------------
    def _format_value(v):
        if isinstance(v, (int, float, np.integer, np.floating)):
            return f"{float(v):.4g}"
        return str(v)

    # compute per-section key widths (keeps alignment within each section)
    blocks: List[str] = []
    for title, d in sections:
        keys = list(d.keys())
        if sort_keys:
            keys = sorted(keys, key=lambda x: str(x))

        key_width = max((len(str(k)) for k in keys), default=0)

        # title
        if title:
            if title_style == "underline":
                blocks.append(str(title))
                blocks.append("-" * len(str(title)))
            else:
                blocks.append(str(title))

        # body
        for k in keys:
            blocks.append(f"{str(k):<{key_width}} : {_format_value(d[k])}")

        # section spacing
        blocks.extend([""] * max(0, int(section_spacing)))

    # trim trailing blank lines
    while blocks and blocks[-1] == "":
        blocks.pop()

    text_str = "\n".join(blocks) if blocks else ""

    # ------------------------- place + fit -------------------------
    text = ax.text(
        loc[0],
        loc[1],
        text_str,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=max_fontsize,
        family=family,
        linespacing=line_spacing,
    )

    fig = ax.figure

    # Ensure renderer exists (esp. in some backends)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ax_bbox = ax.get_window_extent(renderer=renderer)

    # Reduce font size until bbox fits within axes bbox
    for fs in range(int(max_fontsize), int(min_fontsize) - 1, -1):
        text.set_fontsize(fs)
        fig.canvas.draw()
        bbox = text.get_window_extent(renderer=renderer)

        if ax_bbox.contains(bbox.x0, bbox.y0) and ax_bbox.contains(bbox.x1, bbox.y1):
            break

    return text
