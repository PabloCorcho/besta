import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

def draw_dict_in_axes(ax, info: dict,
                      loc=(0.02, 0.98),
                      max_fontsize=14,
                      min_fontsize=6,
                      family="monospace",
                      ha="left",
                      va="top",
                      line_spacing=1.0):
    """
    Render a dictionary as readable text inside a matplotlib Axes,
    automatically adjusting font size so the text fits within the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    info : dict
        Dictionary with numerical (or printable) values.
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

    Returns
    -------
    text : matplotlib.text.Text
        The text artist.
    """

    # Format dictionary into aligned text
    key_width = max(len(str(k)) for k in info)
    lines = [
        f"{k:<{key_width}} : {v:.4g}" if isinstance(v, (int, float)) else f"{k:<{key_width}} : {v}"
        for k, v in info.items()
    ]
    text_str = "\n".join(lines)

    # Initial draw with max font size
    text = ax.text(
        loc[0], loc[1], text_str,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=max_fontsize,
        family=family,
        linespacing=line_spacing
    )

    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    # Axes bounding box in display coordinates
    ax_bbox = ax.get_window_extent(renderer=renderer)

    # Iteratively reduce font size until it fits
    for fs in range(max_fontsize, min_fontsize - 1, -1):
        text.set_fontsize(fs)
        fig.canvas.draw_idle()
        bbox = text.get_window_extent(renderer=renderer)

        if ax_bbox.contains(bbox.x0, bbox.y0) and ax_bbox.contains(bbox.x1, bbox.y1):
            break

    return text
