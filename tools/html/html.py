"""
Our stock HTML template is bootstrap 4, jQuery, zscript

Typical Usage under Flask:

    def scriptblock_root():
        return '''
            $(document).ready(function() {
                let main = z("div",
                    z("h1", "Cool Title"),
                );
                $("#main").html(main);
            });
        '''

    def reply(blocks, code=200):
        if not isinstance(blocks, list):
            blocks = [blocks]
        return html_template(
            "Title of this page",
            blocks,
        ), code

    @app.route("/")
    def root():
        return reply(scriptblock_root())


"""

import json
import os

from plaster.tools.utils.utils import load


def html_template(title, scripts):
    root = os.environ.get("ERISYON_ROOT")
    assert root
    zscript = load(root + "/tools/html/zscript.js")

    newline = "\n"
    return (
        (
            f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{title}</title>
            <link rel="icon" href="data:;base64,iVBORw0KGgo=">
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
            <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
            <link rel="stylesheet" href="https://cdn.datatables.net/1.10.18/css/dataTables.bootstrap4.min.css" crossorigin="anonymous">
            <script src="https://cdn.datatables.net/1.10.18/js/jquery.dataTables.min.js" crossorigin="anonymous"></script>
            <script src="https://cdn.datatables.net/1.10.18/js/dataTables.bootstrap4.min.js" crossorigin="anonymous"></script>
            <script>
            {zscript}
            </script>
            {
                newline.join([f"<script>{newline}{script}{newline}</script>" for script in scripts])
            }
        """
        )
        + (
            """
            <script>
                $(document).ready(function() {
                    setInterval( function interval() {
                        $(".timesince").each((i, item) => {
                            $(item).text(
                                timeSince($(item).data("date")) + " ago"
                            );
                        });
                        return interval;
                    }(), 1000);
                });
            </script>
        </head>
        <body id="main" class="container">
        </body>
        </html>
    """
        )
    )


def scriptblock_data(vars):
    result = []
    for var_name, block in vars.items():
        result += [f"let {var_name}={json.dumps(block)};"]

    return "\n".join(result)
