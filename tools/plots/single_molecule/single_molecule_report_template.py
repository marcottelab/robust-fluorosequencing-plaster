import os

__doc__ = """
HTML Templates for single molecule reports.

Image names follow the convention:
   {run_name}_{field_i}_{channel_i}_{peak_i}.(peak|filt|unfilt).png

"""

# ------------------------------------------------------------
# Report Generators
# ------------------------------------------------------------


# Helpers
def _no_loc(peak_i):
    return (-1, -1)


def _no_stats(peak_i):
    return [
        dict(
            region_i="-",
            start=0,
            end=0,
            min=0,
            max=0,
            mean=0.0,
            std=0.0,
            med=0.0,
            mad=0.0,
        )
    ]


# ------------------------------
# Field Report Generator
# ------------------------------


def generate_single_molecule_report(
    base_path,
    run_name,
    run_channels=[],
    run_fields=[],
    run_peaks=[],
    get_peak_loc=_no_loc,
    get_region_stats=_no_stats,
    index_slug="",
):
    """
    Genereate a field report for each (field_i, channel_i) in the run.

    Callbacks:
    - get_peak_loc - return the peak's location as a tuple (x, y)
    - get_region_stats - return a list of dicts with entries for each region stat
    """

    if index_slug != "":
        index_slug = "_" + str(index_slug)

    kargs = {}
    kargs["run_name"] = run_name

    html_paths = []
    for channel_i in run_channels:
        kargs["channel_i"] = channel_i

        for field_i in run_fields:
            kargs["field_i"] = field_i

            html_path = os.path.join(
                base_path,
                f"{run_name}_field_{field_i}_channel_{channel_i}{index_slug}.html",
            )
            html_paths.append(html_path)
            with open(html_path, "w") as html_out:
                html_out.write(report_header.format(**kargs))

                for row_toggle, peak_i in enumerate(run_peaks):
                    kargs["peak_i"] = peak_i

                    if row_toggle % 2 == 0:
                        html_out.write(row_header)

                    peak_loc = get_peak_loc(peak_i)
                    html_out.write(peak_header.format(peak_loc=peak_loc, **kargs))

                    region_stats = get_region_stats(peak_i)

                    html_out.write(stats_header)
                    for region in region_stats:
                        html_out.write(stats_row.format(**region))
                    html_out.write(stats_footer)

                    html_out.write(peak_images.format(**kargs))

                    html_out.write(peak_footer)

                    if row_toggle % 2 == 1:
                        html_out.write(row_footer)
                # /peak_i

                html_out.write(report_footer)
            # /html_out
        # /field_i
    # /channel_i

    return html_paths


# ------------------------------
# Index Page Generator
# ------------------------------


def get_index_path(base_path, run_name, index_slug=""):
    if index_slug != "":
        index_slug = "_" + str(index_slug)

    return os.path.join(base_path, f"{run_name}{index_slug}_index.html")


def generate_single_molecule_report_index(
    base_path, run_name, html_paths, channels=[], fields=[], peaks=[], index_slug=""
):
    """
    Generate an index page for the report.
    """
    index_path = get_index_path(base_path, run_name, index_slug)

    # Note that callers may have passed in None for some of the kwards

    kargs = {}
    kargs["run_name"] = run_name
    kargs["index_slug"] = index_slug
    kargs["channels"] = "[]" if channels is None else ",".join(str(c) for c in channels)
    kargs["fields"] = "[]" if fields is None else ",".join(str(f) for f in fields)
    kargs["n_peaks"] = 0 if peaks is None else len(peaks)

    with open(index_path, "w") as index_out:
        index_out.write(index_header.format(**kargs))

        for p in html_paths:
            p_rel = p.split("/")[-1]
            p_name = p_rel[: p_rel.find(".html")]
            index_out.write(
                index_report_link.format(
                    single_molecule_report_path=p_rel,
                    single_molecule_report_name=p_name,
                )
            )

        index_out.write(index_footer)

    return index_path


# ------------------------------------------------------------
# Templates
# ------------------------------------------------------------

# Shared
head_includes = """
    <link rel="stylesheet"
          href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
          integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm"
          crossorigin="anonymous">

    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"
            integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN"
            crossorigin="anonymous"></script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"
            integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q"
            crossorigin="anonymous"></script>

    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"
            integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl"
            crossorigin="anonymous"></script>

    <!-- Override Bootstrap's desire to make tables take up 100% of the available width -->
    <style>

table.table-fit {{
    width: auto !important;
    table-layout: auto !important;
}}
table.table-fit thead th, table.table-fit tfoot th {{
    width: auto !important;
}}
table.table-fit tbody td, table.table-fit tfoot td {{
    width: auto !important;
}}

img.img-small {{
    height: 50%;
}}
  </style>
"""


# ------------------------------
# Field Report Templates
# ------------------------------

# The 'header' section contains the head and start of the body.
# Parameters: run_name, field_i
report_header = (
    """
<html>
  <head>
    <title>{run_name}: Field {field_i}</title>
"""
    + head_includes
    + """

</head>

<body>
  <h2 class="ml-4">{run_name}: Field {field_i}</h2>
  <div class="container-fluid">
"""
)

# The start of a new row
row_header = """
    <div class="row mb-4 row-cols-2">
"""

# The title and peak trace image for a peak column
# Parameters: run_name, channel_i, field_i, peak_i, peak_loc
peak_header = """
      <div id="peak-{peak_i}" class="border col-sm-6" align="center">
        <h4>Peak {peak_i} Channel {channel_i} Field {field_i} Loc {peak_loc}</h4>
        <img class="pb-3" src="{run_name}.images/field_{field_i}/{run_name}_{field_i}_{channel_i}_{peak_i}.peak.png"/>
"""

# The table start and header row for the stats table.
stats_header = """
        <table class="table table-bordered table-sm table-fit table-hover my-3">
          <thead>
            <th>Region</th>
            <th>Range</th>
            <th>Length</th>
            <th>Min</th>
            <th>Max</th>
            <th>Mean</th>
            <th>Std</th>
            <th>Median</th>
            <th>MAD</th>
          </thead>
"""

# A row in the stats table
# Parmeters: region_i, start, end, l, min, max, mean, std, med, mad
stats_row = """
          <tr>
            <th scope="row">{region_i:,.0f}</th>
            <td>{start:,.0f}-{end:,.0f}</td>
            <td>{l:,.0f}</td>
            <td>{min:,.0f}</td>
            <td>{max:,.0f}</td>
            <td>{mean:,.0f}</td>
            <td>{std:,.0f}</td>
            <td>{med:,.0f}</td>
            <td>{mad:,.0f}</td>
          </tr>
"""

# The closing table tag for a stat row
stats_footer = """
        </table>
"""

# Div with the filt/unfilt images
# # Parameters: run_name, channel_i, field_i, peak_i
peak_images = """
         <div>
           <img class="img-small" src="{run_name}.images/field_{field_i}/{run_name}_{field_i}_{channel_i}_{peak_i}.filt.png"/>
           &nbsp;&nbsp;
           <img class="img-small" src="{run_name}.images/field_{field_i}/{run_name}_{field_i}_{channel_i}_{peak_i}.unfilt.png"/>
         </div>
"""

# Closing div for the peak column
peak_footer = """
        </div> <!-- /col -->
"""

# Closing div for the row
row_footer = """
      </div> <!-- /row -->
"""

# Closing tags for container/body/html
report_footer = """
    </div> <!-- container -->
  </body>
</html>
"""


# ------------------------------
# Index Page Templates
# ------------------------------

index_header = (
    """
<html>
  <head>
    <title>{run_name} [{index_slug}]: Field Reports</title>
"""
    + head_includes
    + """

</head>

<body>
  <h2 class="pl-3 pt-4 pb-3">{run_name} [{index_slug}] Field Reports</h2>

  <div class="pl-4 pb-3">
    <div><b>Channels:</b> {channels}</div>
    <div><b>Fields:</b> {fields}</div>
    <div><b>Total Peaks:</b> {n_peaks}</div>
  </div>
  <div class="list-group list-group-flush">
"""
)

index_report_link = """
    <a class="list-group-item list-group-item-action pl-4 border-0" href="{single_molecule_report_path}" target="_blank">{single_molecule_report_name}</a>"""

index_footer = """
  </div> <!-- /list-group -->

</body>
</html>
"""

if __name__ == "__main__":

    def _some_stats(peak_i):
        return [
            dict(
                region_i=r,
                start=0,
                end=0,
                l=0,
                min=0,
                max=0,
                mean=0.0,
                std=0.0,
                med=0.0,
                mad=0.0,
            )
            for r in [0, 1, 2, 3]
        ]

    index_slug = "this_is_a_test"

    html_paths = generate_single_molecule_report(
        "",
        "template_test_run",
        [0, 1],  # channels
        range(5),  # fields
        range(11),  # peaks
        get_region_stats=_some_stats,
        index_slug=index_slug,
    )

    index_path = generate_single_molecule_report_index(
        "",
        "template_test_run",
        html_paths,
        channels=[0, 1],
        fields=range(5),
        peaks=range(11),
        index_slug=index_slug,
    )

    print("Wrote", index_path)
