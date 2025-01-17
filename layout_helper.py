import base64
import os

import dash
from dash import dcc, html

# Set the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_standalone_app(
        layout,
        callbacks,
        header_colors,
):
    """Run demo app (tests/dashbio_demos/*/app.py) as standalone app."""
    app = dash.Dash(__name__)
    app.scripts.config.serve_locally = True
    # Handle callback to component with id "fullband-switch"
    app.config['suppress_callback_exceptions'] = True


    # Assign layout
    app.layout = app_page_layout(
        page_layout=layout(),

        standalone=True,
        **header_colors()
    )

    # Register all callbacks
    callbacks(app)

    # return app object
    return app


def app_page_layout(page_layout,
                    bg_color="#506784",
                    font_color="#F3F6FA",
                    **kwargs):  # Accept extra arguments like `text_color`
    return html.Div(
        id='main_page',
        children=[
            dcc.Location(id='url', refresh=False),
            html.Div(
                id='app-page-header',
                children=[
                    html.A(
                        id='app-logo', children=[
                            html.Img(
                                src='data:image/png;base64,{}'.format(
                                    base64.b64encode(
                                        open(
                                            './assets/logo.png', 'rb'
                                        ).read()
                                    ).decode()
                                )
                            )],
                    ),

                ],
                style={
                    'background': bg_color,
                    'color': font_color,
                }
            ),
            html.Div(
                id='app-page-content',
                children=page_layout
            )
        ],
    )
