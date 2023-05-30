import math
import os
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from IPython.display import HTML, display
from matplotlib import cm
from tqdm.auto import tqdm

from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, XY
from plaster.tools.plots.single_molecule.single_molecule_report_template import (
    generate_single_molecule_report,
    generate_single_molecule_report_index,
    get_index_path,
)

__doc__ = """
Single Molecule Reports are static HTML reports that include detailed information for some or all peaks in a run. This inludes peak traces, filtered/unfiltered images for each  timepoint in the trace, and deteected step regions, if present. Reports are organized by field, with an index page that contains links to the individual field reports.

Multiple single molecule reports can be generated to include different collections of fields, channels, and peaks. A full report may include thousands of peaks per field, reports for specific collections of samples help keep the size of the pages to a reasonable size when only a subset of the peaks are of interest. Reports are denoted using an "index_slug" provided by the caller.

Images (plots, filt/unfilt peak images) are generated once per run and shared across reports.

Run names are used in the file names and sub directories to make it clear what the run is if links are shared.

Reports live in a directory under run_path with the following structure:

# Top level report directory
single_molecule_report/

  # Internal index of all the reports generated for this run
  single_molecule_reports.yaml

  # HTML index for a report - contains links to all field reports
  {run_name}[_{index_slug}]_index.html

  # Field reports for each field/channel in the report.
  {run_name}_field_{f}_channel_{c}[_{index_slug}].html

  # Image directory
  {run_name}.images/

     # Images for each field
     field_{f}/
       # Peak images
       {run_name}_{f}_{c}_{p}.[peak, filt, unfilt].png

Report are generated using the following funcion (this really the only public function in this module):

def run_single_molecule_report(
    run_name,
    sigproc_results,
    regions_d=Nonef,
    fields=None,
    channels=None,
    peaks=None,
    peak_limit=None,
    run_type="edman",
    run_path="",
    index_slug="",
    regen_html=False,
    regen_plots=False,
    regen_images=False,
    auto_detect_run_type=False,
    edman_limit=25,
)

Note that it's currently decoupled from the JobResult class. This gives the caller a little more flexibility for running ad hoc reports from the command line.

Developer note: All this module does is generate images and HTML files. Some possible future improvements:

- Generating filt/unfilt images is the most expensive part of this. It's partially optimized, but could be better.
  (option 1) Generate peak images from sigproc or another upstream process that manipulates the full field images
  (option 2) switch to a javascript viewer that loads the field images and uses JS to subsample as peaks are viewed. Could allow for more local context

- Switch to a proper HTML templating system such as jinja.

- Resist the temptation to move this to a dynamic serivce. Generating thousands of images for each page view will always be slow. Serving up existing images is faster.

- Similarly, resist the urge to use React or Bokeh for these pages. There are simply too many images per page and both those frameworks will crumble. Keep It Simple.

"""


# Paths and names

# In the functions below:
#   - 'base_path' refers to the base path for the single_molecule_report directory
#   - 'run_path' refers to the directory for the run
#   - In general, the caller should get these from the job
REGIONS_PATH = "step_regions/{run_name}.regions.pkl"

REPORT_DIR = "single_molecule_report"
INDEX_YAML = "single_molecule_reports.yaml"

MANIFEST_YAML = "job_manifest.yaml"

# Plot Parameters
IMGS_PER_ROW = 10
FIG_WIDTH = 6.0
ROW_HEIGHT = 0.5
IMG_RADIUS = 10
PADDING = 3

DPI = 100

# Image helpders
square_radius = IMG_RADIUS
square_mask = imops.generate_square_mask(square_radius + 1, True)

# Color map for images
viridis = cm.get_cmap("viridis", 256)
viridis.set_under("w")  # anything under vmin should be white


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _get_run_name(run_path):
    """
    Placeholder until we have run/sample names stored in the job_manifest.

    For now, use the parent directory name.
    """
    return run_path.split("/")[-1]


# ------------------------------
# Config/Index Files
# ------------------------------


def load_report_index(base_path):
    index_path = os.path.join(base_path, INDEX_YAML)

    # Create it if it doesn't already exist
    if not os.path.exists(index_path):
        idx = {}
        save_report_index(base_path, idx)
    else:
        with open(index_path) as f:
            idx = yaml.load(f, Loader=yaml.FullLoader)

    return idx


def save_report_index(base_path, report_index):
    index_path = os.path.join(base_path, INDEX_YAML)

    with open(index_path, "w") as f:
        yaml.dump(report_index, f)
    return


# ------------------------------
# Notebook HTML Helpers
# ------------------------------


def single_molecule_report_list_html(run_path, html_root):
    ri_path = os.path.join(run_path, REPORT_DIR)
    report_index = load_report_index(ri_path)

    if html_root[-1] != "/":
        html_root += "/"

    display(HTML("<h2>Single Molecule Reports</h2>"))

    display(HTML("<ul>"))

    for report_path in report_index:
        r_html = report_path.split("/")[-1]
        r_path = html_root + REPORT_DIR + "/" + r_html
        r_name = os.path.splitext(r_html)[0]
        display(HTML(f'  <li><a href="{r_path}" target="_blank">{r_name}</a></li>'))

    display(HTML("</ul>"))

    return


# ------------------------------
# Image Helpers
# ------------------------------


class _FieldImages:
    """
    Local storage for field images/stats
    """

    def __init__(self, filt_im, unfilt_im, zero_min=True):
        self.filt_im = filt_im
        self.unfilt_im = unfilt_im

        self.avg = np.average(unfilt_im)
        self.std = np.std(unfilt_im)

        self.zero_min = zero_min
        if zero_min:
            self.im_min = 0.0
        else:
            self.im_min = min(np.min(unfilt_im), np.min(filt_im))

        self.im_max = self.avg + self.std * 4

        return

    def _thumb(self, src, channel_i, cycle_i, loc):
        if False:
            # imops.extract_with_mask doesn't bounds check and fails
            # for peaks near the border.
            t = imops.extract_with_mask(
                src[channel_i, cycle_i], square_mask, loc=loc, center=True
            )
        else:
            im = src[channel_i, cycle_i]
            x1 = max(0, loc.x - IMG_RADIUS - 1)
            x2 = min(im.shape[1], loc.x + IMG_RADIUS + 1)

            y1 = max(0, loc.y - IMG_RADIUS - 1)
            y2 = min(im.shape[0], loc.y + IMG_RADIUS + 1)

            t = im[y1:y2, x1:x2]

        if self.zero_min:
            # Use 15 to keep the color map from assigning low values to white (not sure why it does that)
            t[t < 15.0] = 15.0

        return t

    def filt_thumb(self, *args):
        return self._thumb(self.filt_im, *args)

    def unfilt_thumb(self, *args):
        return self._thumb(self.unfilt_im, *args)


def im_paste(tar, src, loc):
    """
    Paste src at tar:loc

    (basically imops.accum_inplace without the accum part)
    """
    tar[loc.y : (loc.y + src.shape[0]), loc.x : (loc.x + src.shape[1])] = src
    return


def _extend_trace(x, extend):
    """
    Supersample a trace by adding 'extend' points for each point.
    """
    n_cycles = len(x)
    extended_trace = np.zeros(n_cycles * extend)
    for cycle in range(n_cycles):
        frame_start = cycle * extend
        frame_end = frame_start + extend
        extended_trace[frame_start:frame_end] = x[cycle]
    return extended_trace


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------


def _setup_peak_ticks(peak_trace, ax_peak, ax_hist):
    # Create 5 major ticks - round to the nearest 1000
    v5 = np.max(peak_trace) / 5.0
    tick_spaces = int(round(v5, -3))  # int(v5 - v5 % 1000)
    y_lim = np.max(peak_trace)

    if tick_spaces > 0:
        if np.max(peak_trace) > tick_spaces * 5:
            y_ticks = range(0, tick_spaces * 7, tick_spaces)
            y_lim = tick_spaces * 6
        else:
            y_ticks = range(0, tick_spaces * 6, tick_spaces)
            y_lim = tick_spaces * 5

        ax_peak.set_yticks(y_ticks)
        ax_hist.set_yticks(y_ticks)

        if len(peak_trace) < 16:
            ax_peak.set_xticks(range(0, len(peak_trace)))
        else:
            ax_peak.set_xticks(range(0, len(peak_trace), 5))

        ax_peak.set_ylim([0, y_lim])
        ax_hist.set_ylim([0, y_lim])
        ax_peak.set_xlim([0, len(peak_trace)])

    return y_lim


def _peak_trace(
    results,
    df,
    sig,
    field_ims,
    peak_i,
    regions_df=None,
    channel_i=0,
    extend_trace=1,
    image_base_path="",
    single_row=True,
):
    """
    Create a plot for a peak trace:
      - line plot for the main trace
      - red bars for region boundaries
      - histogram on the right side
    """
    peak_trace = sig[peak_i, channel_i, :]

    if regions_df is None:
        regions = [0, len(peak_trace)]
    else:
        p_filt = regions_df["peak_i"] == peak_i
        c_filt = regions_df["channel_i"] == channel_i
        regions = regions_df[p_filt & c_filt]["start"].values

    if extend_trace > 1:
        peak_trace = _extend_trace(peak_trace, extend_trace)
        regions *= extend_trace

    f, (ax_peak, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(7, 1.5),
        sharey=True,
        gridspec_kw={
            "width_ratios": (7, 1),
            "wspace": 0.0,
            "left": 0.1,
            "right": 0.98,
            "bottom": 0.15,
        },
    )

    y_lim = _setup_peak_ticks(peak_trace, ax_peak, ax_hist)

    ax_peak.plot(peak_trace, color="dodgerblue")
    ax_peak.bar(x=regions, height=y_lim, color="red", width=0.2, alpha=0.5)
    ax_hist.hist(peak_trace, bins=20, orientation="horizontal")

    ax_peak.set_axisbelow(True)
    ax_hist.set_axisbelow(True)
    ax_peak.grid(color="lightgray")
    ax_hist.grid(color="lightgray")

    plt.savefig(image_base_path + ".peak.png", dpi=DPI)
    plt.close()

    return


def _peak_images(
    results,
    df,
    sig,
    field_ims,
    peak_i,
    regions_df=None,
    channel_i=0,
    extend_trace=1,
    image_base_path="",
    single_row=True,
):
    """
    Create images that include all individual peak images for both filtered
    and unfiltered source images.
    """
    # Get data pointers
    peak_records = df[df.peak_i == peak_i]
    field_i = int(peak_records.iloc[0].field_i)

    filt_axs = []
    unfilt_axs = []

    # Setup the image grid
    n_img_rows = math.ceil(results.n_cycles / IMGS_PER_ROW)
    height = ROW_HEIGHT * (4 + n_img_rows * 2 + 1)

    if single_row:
        grid_width = results.n_cycles
    else:
        grid_width = IMGS_PER_ROW

    thumb_hw = HW((square_mask.shape[0], square_mask.shape[1]))
    thumb_im_hw = HW(
        (
            (square_mask.shape[0] + PADDING) * n_img_rows,
            (square_mask.shape[1] + PADDING) * grid_width,
        )
    )

    # Create empty images and set a background below the lowest value
    filt_peak_ims = np.zeros(thumb_im_hw) + (field_ims.im_min - 1.0)
    unfilt_peak_ims = np.zeros(thumb_im_hw) + (field_ims.im_min - 1.0)

    # Extract the images for each cycle and paste them into the master images
    cycle_i = 0
    for row in range(n_img_rows):
        for col in range(grid_width):
            if cycle_i >= results.n_cycles:
                continue

            cycle_rec = peak_records[peak_records.cycle_i == cycle_i].iloc[0]
            peak_loc = XY(cycle_rec.aln_x, cycle_rec.aln_y)

            peak_filt_im = field_ims.filt_thumb(channel_i, cycle_i, peak_loc)
            peak_unfilt_im = field_ims.unfilt_thumb(channel_i, cycle_i, peak_loc)

            loc = XY((thumb_hw.w + PADDING) * col, (thumb_hw.h + PADDING) * row)

            im_paste(filt_peak_ims, peak_filt_im, loc)
            im_paste(unfilt_peak_ims, peak_unfilt_im, loc)

            cycle_i += 1
        # for col
    # for row

    # Save the images
    for im, label in [(filt_peak_ims, ".filt"), (unfilt_peak_ims, ".unfilt")]:
        f, ax = plt.subplots(1, 1)
        ax.axis("off")
        ax.imshow(im, cmap=viridis, vmin=field_ims.im_min, vmax=field_ims.im_max)
        plt.savefig(
            image_base_path + label + ".png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=DPI * 1.3,
        )
        plt.close()

    return


def peak_plots(
    results,
    df,
    sig,
    field_ims,
    peak_i,
    regions_df=None,
    channel_i=0,
    extend_trace=1,
    image_base_path="",
    single_row=True,
    regen_plots=False,
    regen_images=False,
):
    """
    Create a plot with the intensity trace and peak images.

    - Trace Plot
    - Background subtracted peak images
    - Raw peak images

    Plots are saved to:
    - {image_base_path}.peak.png
    - {image_base_path}.filt.png
    - {image_base_path}.unfilt.png

    If single_row is True, each set of images is in a single row,
    otherwise, there are 10 images per row.

    If extend_trace is > 1, each point in the trace is replicated that many times

    Use regen_[plots,images] to force images to be regenerated, otherwise
    only create them if they don't already exist.
    """

    peak_png = image_base_path + ".peak.png"
    if regen_plots or not os.path.exists(peak_png):
        _peak_trace(
            results,
            df,
            sig,
            field_ims,
            peak_i,
            regions_df=regions_df,
            channel_i=channel_i,
            extend_trace=extend_trace,
            image_base_path=image_base_path,
            single_row=single_row,
        )

    filt_png = image_base_path + ".filt" + ".png"
    if regen_images or not os.path.exists(filt_png):
        _peak_images(
            results,
            df,
            sig,
            field_ims,
            peak_i,
            regions_df=regions_df,
            channel_i=channel_i,
            extend_trace=extend_trace,
            image_base_path=image_base_path,
            single_row=single_row,
        )

    return


# ------------------------------------------------------------
# Report Generators
# ------------------------------------------------------------


def field_single_molecule_report(
    results,
    df,
    field_i,
    run_name,
    regions_df=None,
    channels=None,
    peaks=None,
    peak_limit=None,
    extend_trace=5,
    single_row=True,
    base_path="",
    index_slug="",
    status_bar=None,
    regen_plots=False,
    regen_images=False,
):
    """
    Generate a field report for a run. For each channel in the field, generate
    the annotated peak traces, peak spot images, and an HTML file that displays
    them along with region stats.

    If peaks is present, filter the peaks to only include those in the list.
    If peak_limit is set, only process that many peaks in each field.
    peak_limit is not used if peaks are explicitly set.

    Index slug is added to the end of the HTML file, preceded by an underscore

    Use peaks + index_slug to create HTML files with specific sets of peaks after
    the images have been created.
    """
    # Paths and file names
    relative_image_path = run_name + ".images"
    base_image_path = os.path.join(base_path, relative_image_path)
    field_image_path = os.path.join(base_image_path, f"field_{field_i}")
    image_base_file_name = "{run_name}_{field_i}_{channel_i}_{peak_i}"

    # Create the directories
    try:
        os.makedirs(field_image_path)
    except FileExistsError:
        # print(f'Warning: {field_image_path} already exists. Possibly overwriting existing images.')
        pass

    # Get the Data
    field_df = df[df.field_i == field_i]
    sig = results.sig()

    # Setup the iterators
    run_channels = range(results.n_channels)
    field_peaks = field_df.peak_i.unique()
    field_peaks.sort()  # just in case...

    if channels is not None:
        run_channels = channels

    if peaks is not None:
        # Filter the peaks
        field_peaks = [peak_i for peak_i in field_peaks if peak_i in peaks]
    elif peak_limit is not None:
        # Or limit the number of peaks
        field_peaks = field_peaks[:peak_limit]

    # Create the images for each peak
    # Get the full images
    filt_im = results.aln_filt_chcy_ims(field_i)
    unfilt_im = results.aln_unfilt_chcy_ims(field_i)
    field_ims = _FieldImages(filt_im, unfilt_im)

    for channel_i in run_channels:
        status_bar.set_description(
            f"Processing Field {field_i}, Channel {channel_i} ({len(field_peaks)} peaks"
        )

        for peak_i in field_peaks:
            status_bar.update(1)

            # Setup paths
            peak_base_file_name = image_base_file_name.format(
                run_name=run_name,
                field_i=field_i,
                channel_i=channel_i,
                peak_i=peak_i,
            )
            full_peak_png_path = os.path.join(field_image_path, peak_base_file_name)

            # Create the images
            peak_plots(
                results,
                df,
                sig,
                field_ims,
                peak_i,
                regions_df=regions_df,
                extend_trace=extend_trace,
                single_row=single_row,
                image_base_path=full_peak_png_path,
                regen_plots=regen_plots,
                regen_images=regen_images,
            )
        # /peak_i
    # /channel_i

    # Callbacks used by generate field report
    def _get_peak_loc(peak_i):
        peak_records = df[df.peak_i == peak_i]
        cycle_rec = peak_records[peak_records.cycle_i == 0].iloc[0]
        return XY(cycle_rec.aln_x, cycle_rec.aln_y)

    def _get_region_stats(peak_i):
        if regions_df is None:
            return []
        return regions_df[regions_df.peak_i == peak_i].to_dict(orient="records")

    # Create the HTML page
    status_bar.set_description(f"Generating HTML for field {field_i}...")
    status_bar.update(1)
    channel_html_paths = generate_single_molecule_report(
        base_path,
        run_name,
        run_channels,
        run_fields=[field_i],
        run_peaks=field_peaks,
        get_peak_loc=_get_peak_loc,
        get_region_stats=_get_region_stats,
        index_slug=index_slug,
    )

    return channel_html_paths


def run_single_molecule_report(
    run_name,
    sigproc_results,
    regions_df=None,
    fields=None,
    channels=None,
    peaks=None,
    peak_limit=None,
    run_type="edman",
    run_path="",
    index_slug="",
    regen_html=False,
    regen_plots=False,
    regen_images=False,
    auto_detect_run_type=False,
    edman_limit=25,
):
    """
    Create a set of field reports for a run.

    Required Parameters:

    run_name : the name of the run. Typically this is jobs_folder/{run_name}
    sigproc_results : the SigprocV2Results that contains the peaks/images for the report

    Optional Parameters:

    run_path : the parent path for the 'single_molecule_report' directory
    index_slug : a unique id for this particular report

    regions_df : a Pandas datafrom generated by step_regions for addtion region annotations

    fields : a list of fields to include the report. Default is all.
    channels : a list of channls to include in the report. Default is all.
    peaks : a list of peaks to include in the report. Default is all.
    peak_limit : an integer limiting how many peaks to include from each field (only used when peaks=None)

    run_type : "edman" or "bleach" (default is "edman"). Edman runs extend the traces in the plots (5 ticks = 1 cycle).
    auto_detect_run_type : If true, make a best guess as to the run type using edman limit. Use this for auto-generated reports.
    edman_limit : the max cycles to consider a run as edman for auto detection. Default is 25.

    regen_html : Set this to True to regenerate the HTML files if a run with the same index_slug already exists. If this is False and a run already exists with the same slug, no html, plots, or images are generated

    regen_plots, regen_images : Set these to True to regenerate the plots and images. Only necessary if features were added to either or sigproc changed the source traces/images.
    """

    # Setup the report directory
    base_path = os.path.join(run_path, REPORT_DIR)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # It's easy to forget the correct run type argument
    if sigproc_results.n_cycles > edman_limit and run_type == "edman":
        if auto_detect_run_type:
            display(HTML("Autodetect: Photobleach run"))
            run_type = "bleach"
        else:
            response = input(
                "It looks like this is a photobleach run. Continue with Edman settings? [y/N] "
            )
            if response not in ["y", "Y"]:
                return []

    if sigproc_results.n_cycles <= edman_limit and run_type != "edman":
        if auto_detect_run_type:
            display(HTML("Autodetect: Edman run"))
            run_type = "edman"
        else:
            response = input(
                "It looks like this is a Edman run. Continue with photobleach settings? [y/N] "
            )
            if response not in ["y", "Y"]:
                return []

    # Set edman/bleach parameters
    if run_type == "edman":
        edman = True
        extend_trace = 5
        single_row = True
    else:
        edman = False
        extend_trace = 1
        single_row = False

    # Set the report paths
    index_path = get_index_path(base_path, run_name, index_slug)
    report_index = load_report_index(base_path)

    # If regen_html is False and this report already exists, bail
    if not regen_html and index_path in report_index:
        display(
            HTML(f"Run already exists (set regen_html=True to override): {index_path}")
        )
        return []

    # Extract the dataframe for the sigproc results and load the dataframe for the region results.
    df = sigproc_results.fields__n_peaks__peaks()

    if fields is None:
        fields = range(sigproc_results.n_fields)

    if channels is None:
        channels = range(sigproc_results.n_channels)

    n_fields = len(fields)
    n_channels = len(channels)

    # Note that peaks are handled differntly than fields and channels since
    # they can be set explicitly using a list, implicitly using None,
    # or indirectly using peak_limit.
    n_peaks = len(sigproc_results.peaks()) if peaks is None else len(peaks)

    # Setup the total ticks for the status bar
    total = 0

    # ...counts for images
    if peak_limit is not None:
        total = peak_limit * n_channels * n_fields
    else:
        total = n_peaks * n_channels

    # ...counts for field html files
    total += n_fields

    # Generate the report for each field
    field_status = tqdm(total=total, desc="Initializing field report...")
    field_html_paths = []

    for field_i in fields:
        channel_html_paths = field_single_molecule_report(
            sigproc_results,
            df,
            field_i,
            run_name,
            regions_df=regions_df,
            channels=channels,
            peaks=peaks,
            peak_limit=peak_limit,
            extend_trace=extend_trace,
            single_row=single_row,
            base_path=base_path,
            status_bar=field_status,
            index_slug=index_slug,
            regen_plots=regen_plots,
            regen_images=regen_images,
        )

        field_html_paths += channel_html_paths

    #  Generate the Report Index HTML File
    field_status.set_description("Saving report index...")

    index_params = dict(
        channels=list(channels) if channels is not None else [],
        fields=list(fields) if fields is not None else [],
        peaks=list(peaks) if peaks is not None else [],
        index_slug=index_slug,
    )

    index_path = generate_single_molecule_report_index(
        base_path, run_name, field_html_paths, **index_params
    )

    # Update the Report Index YAML with the metadata for this report
    report_index[index_path] = index_params
    report_index[index_path]["reports"] = field_html_paths
    report_index[index_path]["index_slug"] = index_slug

    save_report_index(base_path, report_index)

    field_status.set_description("Done")
    field_status.close()

    return field_html_paths


if __name__ == "__main__":
    # Command line test version/development test version
    import sys

    JOBS_FOLDER = "/erisyon/internal/jobs_folder"
    SIGPROC_PATH = "sigproc_v2/plaster_output/sigproc_v2"

    if len(sys.argv) == 1:
        print(f"Usage: python {sys.argv[0]} [bleach] run_names...")
        sys.exit(0)

    args = sys.argv[1:]

    if len(args) > 1 and args[0] == "bleach":
        run_type = "bleach"
        args = args[1:]
    else:
        run_type = "edman"

    # Used for debugging HTML generation
    regen_plots = True
    regen_images = True

    for run_name in args:

        # Assume this will be exected inside the container and set paths accordingly.
        base_run_path = os.path.join(JOBS_FOLDER, run_name)
        sig_path = os.path.join(base_run_path, SIGPROC_PATH)
        regions_path = os.path.join(
            base_run_path, REGIONS_PATH.format(run_name=run_name)
        )

        # Load the sigproc results for the run
        results = SigprocV2Result(sig_path)

        print("Processing ", run_name)
        print("    ", sig_path)

        # Extract the dataframe for the sigproc results and load
        # the dataframe for the region results.
        df = results.fields__n_peaks__peaks()
        if os.path.exists(regions_path):
            with open(regions_path, "rb") as f:
                regions_df = pickle.load(f)
        else:
            print("Regions dataframe not found. Report will not contain region details")
            regions_df = None

        # Testing parameters
        fields = [0]
        channels = [0, 1]
        peaks = list(range(10))

        run_single_molecule_report(
            run_name,
            results,
            regions_df,
            fields=fields,
            channels=channels,
            peaks=peaks,
            run_path=base_run_path,
            run_type=run_type,
            index_slug="cli_test",
            regen_plots=True,
            regen_images=True,
        )
