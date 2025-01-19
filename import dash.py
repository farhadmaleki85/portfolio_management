import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)

# Fetch data
def fetch_sp500_data():
    """
    Fetch the list of S&P 500 companies and their metadata (symbol, security name, and sector).

    Returns:
        tuple: DataFrame of S&P 500 data, list of tickers, dropdown options, and unique sectors.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    sp500_table = sp500_table[['Symbol', 'Security', 'GICS Sector']]
    sp500_table.loc[:, 'Security'] = sp500_table['Security'] + ' (' + sp500_table['Symbol'] + ')'

    # Extract required data
    sp500_tickers = sp500_table['Symbol'].tolist()
    sp500_options = [{'label': row['Security'], 'value': row['Symbol']} for _, row in sp500_table.iterrows()]
    sectors = sp500_table['GICS Sector'].unique()

    return sp500_table, sp500_tickers, sp500_options, sectors

# Get data for S&P 500
sp500_df, sp500_tickers, sp500_options, sectors = fetch_sp500_data()

# Create layout
app.layout = html.Div([
    # Store for the data
    dcc.Store(id='cache-store', storage_type='memory'),  # Store the data in memory

    # Placeholder for dropdown of valid tickers
    dcc.Dropdown(id='tickers-dropdown', options=sp500_options, placeholder="Select a Ticker"),

    # Placeholder for any graph or additional info
    html.Div(id='output-container')
])

# Function to download data for all tickers
def download_all_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

# Callback to download and store data when the app starts or as needed
@app.callback(
    Output('cache-store', 'data'),
    [Input('cache-store', 'data')],
    prevent_initial_call=True  # This ensures the callback is not triggered at first load
)
def download_data_if_needed(cache_data):
    if cache_data is None:
        # Use the actual S&P 500 tickers list
        tickers = sp500_tickers
        start_date = '2022-01-01'
        end_date = '2024-01-01'

        # Download data for all tickers
        data = download_all_data(tickers, start_date, end_date)
        
        # Convert the data to a format suitable for storing in dcc.Store
        cache_data = data.to_dict()  # Store as a dictionary of tickers and their data

    return cache_data

# Callback to update the dropdown options based on the downloaded data
@app.callback(
    Output('tickers-dropdown', 'options'),
    [Input('cache-store', 'data')]
)
def update_dropdown_options(cache_data):
    if cache_data is None:
        return []

    # Create dropdown options from the cached data
    options = [{'label': ticker, 'value': ticker} for ticker in cache_data.keys()]
    return options

# Callback to handle user selection and display data or graph
@app.callback(
    Output('output-container', 'children'),
    [Input('tickers-dropdown', 'value')],
    [State('cache-store', 'data')]
)
def display_ticker_data(selected_ticker, cache_data):
    if selected_ticker is None or cache_data is None:
        return "Please select a ticker."

    # Get the data for the selected ticker from the cached data
    ticker_data = cache_data.get(selected_ticker)

    # You can display the data as a table, plot, or any other format
    return html.Div([
        html.H3(f"Data for {selected_ticker}"),
        html.Pre(str(ticker_data.tail()))  # Just show the last few rows of data for simplicity
    ])

# To run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
