import binascii
import os
import pathlib
import random
import re
import typing
from contextlib import contextmanager

import numpy as np
import numpy.typing

from plaster.tools.schema import check
from plaster.tools.utils import utils


def dropdown(df, description, value):
    import ipywidgets as widgets  # Defer slow import

    return widgets.Dropdown(
        options=sorted(df.columns), description=description, value=value
    )


def tooltips(df):
    tooltips = [(key, f"@{key}") for key in sorted(df.columns)]
    tooltips += [
        ("peak", "$index"),
        (
            "(x,y)",
            "($x, $y)<style>.bk-tooltip>div:not(:first-child) {display:none;}</style>",
        ),
    ]
    return tooltips


def restart_kernel():
    from IPython.display import display_html  # Defer slow imports

    display_html("<script>Jupyter.notebook.kernel.restart()</script>", raw=True)


def _css_for_collapsible():
    from IPython.core.display import HTML, display  # Defer slow imports

    display(
        HTML(
            """
            <style>
                .wrap-collabsible {
                  margin-bottom: 0.2rem 0;
                }

                input[type='checkbox'] {
                  display: none;
                }

                .lbl-toggle {
                  display: block;
                  font-weight: bold;
                  font-size: 130%;
                  //text-transform: uppercase;
                  cursor: pointer;
                  border-radius: 7px;
                  transition: all 0.25s ease-out;
                }

                .lbl-toggle::before {
                  content: ' ';
                  display: inline-block;
                  border-top: 5px solid transparent;
                  border-bottom: 5px solid transparent;
                  border-left: 5px solid currentColor;
                  vertical-align: middle;
                  margin-right: .7rem;
                  transform: translateY(-2px);
                  transition: transform .2s ease-out;
                }

                .toggle:checked + .lbl-toggle::before {
                  transform: rotate(90deg) translateX(-3px);
                }

                .collapsible-content {
                  max-height: 0px;
                  overflow: hidden;
                  transition: max-height .25s ease-in-out;
                }

                .toggle:checked + .lbl-toggle + .collapsible-content {
                  max-height: 10000000350px;
                }

                .toggle:checked + .lbl-toggle {
                  border-bottom-right-radius: 0;
                  border-bottom-left-radius: 0;
                }

                .collapsible-content .content-inner {
                  border: 2px solid rgba(0,0,0,0.2);
                  border-radius: 6px;
                  padding: .5rem 1rem;
                }
            </style>
        """
        )
    )


@contextmanager
def wrap_in_collapsible(header):
    from IPython.core.display import HTML, display  # Defer slow imports
    from IPython.utils.capture import capture_output  # Defer slow imports

    def _id():
        return binascii.b2a_hex(os.urandom(16)).decode("ascii")

    with capture_output(stdout=True, stderr=True, display=True) as captured:
        yield captured

    # captured now has the content which will be rendered into .output
    # divs in the display() call below. Before and after that special
    # sentinel divs are added so that the divs between those can be pulled
    # into a collapsible.
    _css_for_collapsible()

    top_id = _id()
    display(HTML(f"<div id='{top_id}'></div>"))

    for o in captured.outputs:
        o.display()

    bot_id = _id()
    display(HTML(f"<div id='{bot_id}'></div>"))

    # Run some jQuery kung-fu to pull the .output_subareas into the collapsible
    display(
        HTML(
            """
        <script>
            var top0 = $('#"""
            + top_id
            + """').closest('.output_area');
            var bot0 = $('#"""
            + bot_id
            + """').closest('.output_area');
            var foundOutputAreas = $(top0).nextUntil($(bot0));
            var topSubArea = $(top0).find('.output_subarea');
            $(topSubArea).empty();
            $(topSubArea).html(
                "<div class='wrap-collabsible'>" +
                    "<input id='collapsible-"""
            + top_id
            + """' class='toggle' type='checkbox'>" +
                    "<label for='collapsible-"""
            + top_id
            + """' class='lbl-toggle'>"""
            + header
            + """</label>" +
                    "<div class='collapsible-content'>" +
                        "<div class='content-inner'>" +
                        "<p></p>" +
                        "</div>" +
                    "</div>" +
                "</div>"
            );
            var insideOfCollapsable = $(top0).find("p");
            var foundOutputSubAreas = $(foundOutputAreas).find('.output_subarea');
            $(foundOutputSubAreas).detach().appendTo(insideOfCollapsable);
        </script>
    """
        )
    )


"""
Example usage of hd, h

hd("h1", "Some title")

hd("div",
    h("p.some_class.another_class", "paragraph 1"),
    h("p#the_id", "paragraph 2"),
    h(".a_class_on_a_div", "A div"),
)

hd("div",
    h("div",
        h("p", "paragraph 1"),
        h("p", "paragraph 2"),
    ),
    h("div",
        h("p", "paragraph 3")
    )
)
"""


def _h_fmt(_tag):
    id = ""
    classes = ""
    tag = "div"
    for part in re.split(r"([.#][^.#]+)", _tag):
        if part.startswith("#"):
            id = f"{part[1:]}"
        elif part.startswith("."):
            classes += f"{part[1:]} "
        elif part != "":
            tag = part

    return tag, id, classes


def h(tag, *strings):
    tag, id, classes = _h_fmt(tag)
    return f"<{tag} id='{id}' class='{classes}'>{' '.join([str(s) for s in strings])}</{tag}>"


def hd(tag, *strings):
    from IPython.core.display import HTML, display  # Defer slow imports

    display(HTML(h(tag, *strings)))


def md(string):
    from IPython.core.display import Markdown, display  # Defer slow imports

    display(Markdown(string))


def v(vec, prec=2):
    """format a vector"""
    if isinstance(vec, list):
        vec = np.array(vec)
    return ", ".join([f"{i:2.{prec}f}" for i in vec.squeeze()])


def m(mat, prec=2, indent=""):
    """format a matrix"""
    assert mat.ndim == 2
    return "\n".join([indent + v(row, prec) for row in mat])


def pv(vec, prec=2):
    """print a vector"""
    print(v(vec, prec))


def pm(mat, prec=2):
    """print a matrix"""
    print(m(mat, prec))


def title(title):
    md(f"# {title}")


def subtitle(title):
    md(f"### {title}")


def fix_auto_scroll():
    from IPython.core.display import HTML, display  # Defer slow imports

    display(
        HTML(
            """
                <script>
                    var curOutput = $(".jupyter-widgets-output-area", Jupyter.notebook.get_selected_cell().element.get(0));
                    var curOutputChildren = $(curOutput).children(".output");
                    var mut = new MutationObserver(function(mutations) {
                        mutations.forEach(function (mutation) {
                            $(mutation.target).removeClass("output_scroll");
                          });
                    });
                    mut.observe( $(curOutputChildren)[0], {'attributes': true} );
                </script>
            """
        )
    )


def qgrid_mono():
    from IPython.core.display import HTML, display  # Defer slow imports

    display(
        HTML(
            "<style>.slick-cell { font-family: monospace, monospace !important; }</style>"
        )
    )


def css_for_markdown():
    """
    Not yet tested. The idea is to limit the width of markdown.
    """

    from IPython.core.display import HTML, display  # Defer slow imports

    display(
        HTML(
            """
            <style>
                .text_cell {
                    max-width: 300;
                }
            </style>
        """
        )
    )


def explanation(text_or_h):
    from IPython.core.display import HTML, display  # Defer slow imports

    display(
        HTML(
            """
            <style>
                .zfold {
                    display: revert;
                }
                .zfold summary {
                    display: revert;
                }
                .zfold div {
                    margin-left: 1em;
                }
            </style>
        """
        )
    )

    display(
        HTML(
            h(
                "details.zfold",
                h("summary", "Explanation"),
                h("div", h("pre", utils.smart_wrap(text_or_h))),
            )
        )
    )


def movie_pil(
    file_path,
    ims,
    overlay=None,
    _cspan=None,
    _cper=None,
    _size=None,
    _labels=None,
    _frame_duration_ms=250,
    _quality=50,
):
    """
    Render a movie to animated gif format and save it.

    Note: This was converted to use webp which does have
    significantly better space savings but the webp format
    isn't supported by Google Slides, etc which makes it
    much less convenient. (Ironoic since webp is a Google Product)

    _quality for webp of 50% was good although ti does introduce
    some compression artifacts. I don't think quality has meaning
    in GIF mode.

    Optional overlay is RGBA of same size
    """
    check.array_t(ims, ndim=3)

    from PIL import Image, ImageDraw, ImageFont

    if _cspan is not None:
        bot, top = _cspan
    elif _cper is not None:
        bot, top = np.percentile(ims, _cper)
    else:
        bot, top = 0.0, np.percentile(ims, 99.99)

    # For now using the "better than nothing" font
    # I would need to copy a ttf font over to a directory
    # in our tree to be able to use it on a remote machine
    font = ImageFont.load_default()

    pil_ims = []
    for i, im in enumerate(ims):
        _im = np.clip(255 * (im - bot) / (top - bot), a_min=0, a_max=255)
        _im = _im[::-1, :].astype(np.uint8)
        _im = np.repeat(_im[:, :, None], 3, axis=2)
        _im = Image.fromarray(_im, mode="RGB")
        if _size is not None:
            _im = _im.resize((_size, _size), resample=Image.BICUBIC)

        draw = ImageDraw.Draw(_im)
        if _labels is not None:
            draw.text((6, 11), _labels[i], fill="black", font=font)
            draw.text((5, 10), _labels[i], fill="white", font=font)

        if overlay is not None:
            over_im = Image.fromarray(overlay[i][::-1])
            _im.paste(over_im, (0, 0), over_im)

        pil_ims += [_im]

    if _size is None and pil_ims:
        _size = pil_ims[0].size[0]

    pil_ims[0].save(
        fp=file_path,
        # format="WEBP",
        format="GIF",
        # quality=_quality,
        append_images=pil_ims,
        save_all=True,
        duration=_frame_duration_ms,
        loop=0,
    )

    return _size


def movie(ims, **kwargs):
    """
    Render movie and display in notebook
    See movie_pil for options
    """
    from IPython.core.display import HTML, display  # Defer slow imports

    # Add a code to cache bust
    code = random.randint(0, 2e9)
    # file_path = f"./__image_{code}.webp"
    file_path = f"./__image_{code}.gif"
    size = movie_pil(file_path=file_path, ims=ims, **kwargs)
    display(HTML(f"<img src='{file_path}' width='{size}'>"))


def multichannel_movie_pil(
    file_path,
    ims_red,
    ims_green,
    ims_blue,
    _size=None,
    _labels=None,
    _frame_duration_ms=250,
):
    check.array_t(ims_red, ndim=3)
    check.array_t(ims_green, ndim=3)
    check.array_t(ims_blue, ndim=3)

    from PIL import Image, ImageDraw, ImageFont

    dim = ims_red.shape[-2:]
    assert ims_red.shape == ims_green.shape and ims_red.shape == ims_blue.shape

    # For now using the "better than nothing" font
    # I would need to copy a ttf font over to a directory
    # in our tree to be able to use it on a remote machine
    font = ImageFont.load_default()

    pil_ims = []
    for i, (im_r, im_g, im_b) in enumerate(zip(ims_red, ims_green, ims_blue)):
        _im_r = np.clip(255 * im_r, a_min=0, a_max=255)
        _im_r = _im_r[::-1, :].astype(np.uint8)

        _im_g = np.clip(255 * im_g, a_min=0, a_max=255)
        _im_g = _im_g[::-1, :].astype(np.uint8)

        _im_b = np.clip(255 * im_b, a_min=0, a_max=255)
        _im_b = _im_b[::-1, :].astype(np.uint8)

        rgb_im = np.zeros((*dim, 3), dtype=np.uint8)
        rgb_im[:, :, 0] = _im_r
        rgb_im[:, :, 1] = _im_g
        rgb_im[:, :, 2] = _im_b

        _im = Image.fromarray(rgb_im, mode="RGB")
        if _size is not None:
            _im = _im.resize((_size, _size), resample=Image.BICUBIC)

        draw = ImageDraw.Draw(_im)
        if _labels is not None:
            draw.text((6, 11), _labels[i], fill="black", font=font)
            draw.text((5, 10), _labels[i], fill="white", font=font)

        pil_ims += [_im]

    if _size is None and pil_ims:
        _size = pil_ims[0].size[0]

    pil_ims[0].save(
        fp=file_path,
        format="GIF",
        append_images=pil_ims,
        save_all=True,
        duration=_frame_duration_ms,
        loop=0,
    )

    return _size


def normalize(ims, cper=(50, 99.9)):
    bot, top = np.percentile(ims, cper)
    return (ims - bot) / (top - bot)


def normalize_image_stack_against_another(
    image_stack_to_normalize: np.ndarray,
    image_stack_to_normalize_against: np.typing.ArrayLike,
    percentile_limits: typing.Tuple[float, float] = (50, 99.9),
) -> np.ndarray:
    """
    Normalizes image_stack_to_normalize against image_stack_to_normalize_against using percentile_limits to define the lower and upper range.
    I.e., values in image_stack_to_normalize less than or equal to the percentile_limits[0] percentile will become zero;
          values in image_stack_to_normalize greater than or equal to the percentile_limits[1] percentile will become 1;
          and values between these end points will be linearally interpolated.
    """

    bottom_percentile, top_percentile = np.percentile(
        image_stack_to_normalize_against, percentile_limits
    )
    normalized_image_stack = (image_stack_to_normalize - bottom_percentile) / (
        top_percentile - bottom_percentile
    )
    return normalized_image_stack


def multichannel_movie(
    ims_red: np.typing.ArrayLike,
    ims_green: np.typing.ArrayLike,
    ims_blue: np.typing.ArrayLike,
    file_path: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Render movie and display in notebook
    See movie_pil for options
    """
    from IPython.core.display import HTML, display  # Defer slow imports

    if file_path is None:
        # Add a code to cache bust
        code = random.randint(0, 2e9)
        file_path = f"./__image_{code}.gif"

    size = multichannel_movie_pil(
        file_path=file_path,
        ims_red=ims_red,
        ims_green=ims_green,
        ims_blue=ims_blue,
        **kwargs,
    )
    display(HTML(f"<img src='{file_path}' width='{size}'>"))
