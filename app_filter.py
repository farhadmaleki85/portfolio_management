"""
Portfolio Management Application
=================================

This Dash-based application enables users to create, analyze, and optimize investment portfolios. The app provides features such as:

1. Portfolio creation using S&P 500 stocks.
2. Advanced optimization techniques (e.g., Maximum Sharpe Ratio, Minimum Volatility).
3. Asset filtering by sectors.
4. Visualizations for trends, correlations, and allocations.
5. Portfolio performance metrics such as returns, volatility, and Sharpe ratio.

"""

# import functools
import base64
import os

from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import yfinance as yf
import numpy as np
import scipy.optimize as sco
from layout_helper import run_standalone_app

# ======================================
# Data Fetching and Preprocessing
# ======================================

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
sp500_df, sp500_tickers, sp500_options, sectors = fetch_sp500_data()

# Process data
def process_data(selected_tickers, start_date, end_date, interval, optimize_clicks, optimization_methods, risk_free_rate=0.0):
    """
    Process the selected tickers, calculate metrics, and return results.

    Parameters:
        selected_tickers (list): List of selected ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Resampling interval (e.g., 'D', 'W', 'M').
        optimize_clicks (int): Number of times optimization is triggered.
        optimization_methods (list): List of optimization methods to use.
        risk_free_rate (float): Risk-free rate for calculating Sharpe ratio. Default is 0.0.

    Returns:
        dict: Portfolio metrics, optimized weights, and removed assets.
    """
    try:
        # Download data for selected tickers and benchmark
        data = yf.download(selected_tickers, start=start_date, end=end_date)['Close']
        sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Close']
        data = data.dropna()
        sp500_data = sp500_data.dropna()
        # Clean the data using clean_price function        
        clean_data = data.interpolate(method='linear', limit_direction='forward')
        clean_sp500_data = sp500_data.interpolate(method='linear', limit_direction='forward')
        # Resample data
        resampled_data = resample_data(clean_data, interval)
        resampled_sp500_data = resample_data(clean_sp500_data, interval)
        
        # Calculate portfolio metrics
        returns = resampled_data.pct_change().dropna()
        weights = [1 / len(selected_tickers)] * len(selected_tickers)  # Default equal weights
        sp500_returns = resampled_sp500_data.pct_change().dropna()
        
        # Handle portfolio optimization
        removed_assets = []
        if optimize_clicks > 0:
            # Perform optimization
            optimized_weights = optimize_portfolio(selected_tickers, resampled_data, optimization_methods)
            
            # Filter out assets with negligible weights
            filtered_weights = {ticker: weight for ticker, weight in zip(selected_tickers, optimized_weights) if weight >= 0.00001}
            removed_assets = [ticker for ticker, weight in zip(selected_tickers, optimized_weights) if weight < 0.00001]
            
            # Update weights
            weights = [filtered_weights.get(ticker, 0) for ticker in selected_tickers]
        
        # Calculate portfolio metrics
        weighted_returns, mean_return, volatility, sharpe_ratio = calculate_portfolio_metrics(
            resampled_data, selected_tickers, weights, risk_free_rate
        )
        
        # Return all results as a dictionary
        return {
            'clean_data' : clean_data,
            'clean_sp500_data' : clean_sp500_data,
            "resampled_data": resampled_data,
            "resampled_sp500_data": resampled_sp500_data,
            "returns": returns,
            "sp500_returns": sp500_returns,
            "weights": weights,
            "removed_assets": removed_assets,
            "metrics": {
                "weighted_returns": weighted_returns,
                "mean_return": mean_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
            }
        }    


    except Exception as e:
        print(f"Error processing data: {e}")
        return {}
    
# ======================================
# Utility Functions
# ======================================
# Helper functions to calculate portfolio metrics
def calculate_portfolio_metrics(data, tickers, weights, risk_free_rate=0.0):
    """
    Calculate key portfolio metrics such as weighted returns, mean return, volatility, and Sharpe ratio.

    Args:
        data (DataFrame): Historical price data.
        tickers (list): List of asset tickers.
        weights (list): Portfolio weights for each ticker.
        risk_free_rate (float, optional): Risk-free rate for Sharpe ratio calculation. Defaults to 0.0.

    Returns:
        tuple: Weighted returns, mean return, volatility, and Sharpe ratio.
    """
    # Exclude benchmark ticker ('^GSPC') from calculations
    tickers_no_benchmark = [ticker for ticker in tickers if ticker != '^GSPC']
    
    # Ensure weights match the number of tickers (excluding the benchmark)
    if len(weights) != len(tickers_no_benchmark):
        raise ValueError(f"Number of weights ({len(weights)}) does not match the number of selected tickers ({len(tickers_no_benchmark)})")
    
    # Extract data for selected tickers (excluding benchmark)
    data_no_benchmark = data[tickers_no_benchmark]
    
    # Calculate daily returns
    daily_returns = data_no_benchmark.pct_change(fill_method=None).dropna()  # Updated to avoid the FutureWarning
    mean_returns = daily_returns.mean()
    covariance_matrix = daily_returns.cov()
    
    # Calculate portfolio metrics using helper functions
    weighted_returns = (daily_returns * weights).sum(axis=1)
    mean_return = calculate_portfolio_return(weights, mean_returns)
    volatility = calculate_portfolio_volatility(weights, covariance_matrix)  # Fixed the weights issue
    sharpe_ratio = calculate_sharpe_ratio(weights, mean_returns, covariance_matrix, risk_free_rate)
    
    return weighted_returns, mean_return, volatility, sharpe_ratio

def calculate_portfolio_return(weights, mean_returns):
    """Calculate the portfolio's expected return."""
    return np.sum(weights * mean_returns)

def calculate_portfolio_volatility(weights, covariance_matrix):
    """Calculate the portfolio's volatility."""
    # Ensure weights is a NumPy array
    weights = np.array(weights)
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

# Helper function to calculate equally weighted portfolio
def calculate_equally_weighted_portfolio(tickers):
    num_assets = len(tickers)
    return np.full(num_assets, 1.0 / num_assets)

def calculate_sharpe_ratio(weights, mean_returns, covariance_matrix, risk_free_rate=0.0):
    """Calculate the portfolio's Sharpe ratio."""
    portfolio_return = calculate_portfolio_return(weights, mean_returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
    return (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else np.nan

def calculate_omega_ratio(weights, data, threshold=0.0):
    """Calculate the Omega ratio of the portfolio."""
    daily_returns = data.pct_change().dropna()
    portfolio_returns = np.dot(daily_returns, weights)
    gains = portfolio_returns[portfolio_returns > threshold]
    losses = portfolio_returns[portfolio_returns <= threshold]
    return np.sum(gains) / np.abs(np.sum(losses)) if np.sum(losses) != 0 else np.inf

def calculate_erc(weights, covariance_matrix):
    """Calculate the Equal Risk Contribution (ERC) objective."""
    portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
    marginal_risk_contributions = np.dot(covariance_matrix, weights) / portfolio_volatility
    risk_contributions = marginal_risk_contributions * weights / portfolio_volatility
    return np.sum((risk_contributions - 1 / len(weights)) ** 2)  # Minimize deviations


# Modify optimize_portfolio to include equally weighted strategy
def optimize_portfolio(tickers, data, criterion= None, risk_free_rate=0.0):
    """
    Optimize the portfolio based on a specified criterion (e.g., Sharpe ratio, volatility).

    Args:
        tickers (list): List of asset tickers.
        data (DataFrame): Historical price data.
        criterion (str, optional): Optimization criterion. Defaults to None.
        risk_free_rate (float, optional): Risk-free rate for Sharpe ratio calculation. Defaults to 0.0.

    Returns:
        list: Optimized portfolio weights.
    """
    if not tickers:
        raise ValueError("No tickers provided for optimization.")
    
    # If equally weighted strategy is chosen, return equal weights
    if criterion in [None, 'equal_weights']:
        return calculate_equally_weighted_portfolio(tickers)

    # Calculate daily returns and covariance matrix
    daily_returns = data[tickers].pct_change().dropna()
    # mean_returns = daily_returns.mean()
    covariance_matrix = daily_returns.cov()
    num_assets = len(tickers)
    initial_weights = np.full(num_assets, 1.0 / num_assets)  # Equal initial weights
    
    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # Weights sum to 1
    bounds = [(0, 1) for _ in range(num_assets)]  # Weights between 0 and 1

    # Map optimization criterion to objective function
    objective_func_mapping = {
        'sharpe': lambda weights: -calculate_sharpe_ratio(weights, daily_returns.mean(), covariance_matrix),
        'volatility': lambda weights: calculate_portfolio_volatility(weights, covariance_matrix),
        'omega_ratio': lambda weights: -calculate_omega_ratio(weights, data),
        'erc': lambda weights: calculate_erc(weights, covariance_matrix)
    }

    objective_function = objective_func_mapping.get(criterion)
    if not objective_function:
        raise ValueError("Invalid optimization criterion. Choose from 'sharpe', 'volatility', 'omega_ratio', or 'erc'.")

    # Perform optimization
    result = sco.minimize(
        fun=objective_function,
        x0=initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )
    
    optimal_weights = result.x
    return optimal_weights


# Function for conditional styling with background color bins
def discrete_background_color_bins(df, n_bins=5):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    df_max = df.max().max()
    df_min = df.min().min()
    ranges = [((df_max - df_min) * i) + df_min for i in bounds]
    styles = []

    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
        color = 'white' if i > len(bounds) / 2 else 'black'

        for column in df.columns:
            styles.append({
                'if': {
                    'filter_query': (
                        f'{{{column}}} >= {min_bound}' +
                        (f' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })

    return styles

def resample_data(data, interval):
    # Map intervals to resampling codes
    resample_map = {
        '1W': 'W',   # Weekly
        '1M': 'M',   # Monthly
        '1A': 'A',   # Annually
    }

    # Get the resampling code from the map or default to 'D' (daily) if not found
    resample_code = resample_map.get(interval, 'D')

    # Resample and return the data
    return data.resample(resample_code).last()


# ======================================
# Layout and Callbacks
# ======================================

color_palette = [
    'rgb(84,48,5)',
    'rgb(246,232,195)',
    'rgb(0,60,48)'
]
def header_colors():
    return {
        'bg_color': '#232323',
        'font_color': 'white'
    }

def layout():
    """
    Define the layout of the Dash application.

    Returns:
        html.Div: Layout of the application.
    """
    return html.Div(
        id='pfmanager-body', 
        className='app-body',
        children=[
            dcc.Store(id="cached-content", storage_type="memory"),  # Store cached data

            # Hidden div to store the portfolio state (selected tickers)
            html.Div(id='portfolio-state', style={'display': 'none'}),  # Portfolio State

            # Control Tabs
            html.Div(
                id='pfmanager-control-tabs',
                className='control-tabs',
                children=[
                    dcc.Tabs(
                        id='pfmanager-tabs',
                        value='what-is',
                        children=[
                            # About Tab
                            dcc.Tab(
                                label='About',
                                value='what-is',
                                children=html.Div(
                                    className='about-tab-content control-tab',
                                    children=[
                                        html.H4('What is Portfolio Manager?'),
                                        html.P('This app helps create, optimize, and monitor investment portfolios.'),
                                        html.Ul([
                                            html.Li('Portfolio Creation: Build portfolios based on goals and constraints.'),
                                            html.Li('Optimization Tools: Advanced strategies like Omega Ratio and Maximum Sharpe Ratio.'),
                                            html.Li('Asset Coverage: Includes all S&P 500 symbuls.'),
                                            html.Li('Tracking & Insights: Provides analytical metrics such as correlation, average return and volatility for individual assets and selected portfolio.')
                                        ])
                                    ]
                                )
                            ),

                            # Setup Tab
                            dcc.Tab(
                                label='Setup',
                                value='datasets',
                                children=html.Div(
                                    className='control-tab',
                                    children=[

                                        html.Div('Time Frame', className='fullwidth-app-controls-name' ),
                                        dcc.DatePickerRange(
                                            id='date-picker',
                                            start_date='2000-01-01',
                                            end_date='2024-12-31',
                                            display_format='YYYY-MM-DD',
                                            className='custom-date-picker'
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        html.Br(),

                                        # Asset Selection by Sector Dropdown
                                        html.Label("Select an Industry", className='fullwidth-app-controls-name'),
                                        dcc.Dropdown(
                                            id='industry-dropdown',
                                            options=[{'label': sector, 'value': sector} for sector in sectors],
                                            value=['Industrials'],  # Set preset industry value
                                            placeholder="Select an industry",
                                            multi=False,
                                            clearable =False
                                        ),
                                        html.Br(),

                                        html.Label("Select Stocks", className='fullwidth-app-controls-name'),
                                        dcc.Dropdown(
                                            id='ticker-dropdown',
                                            options=[],  # Populated dynamically
                                            value=['MMM', 'AOS', 'ALLE', 'AME', 'ADP'], 
                                            placeholder="Select tickers",
                                            multi=True,
                                            clearable =False

                                        ),
                                        html.Br(),

                                        # html.Div('Add To Portfolio', className='fullwidth-app-controls-name'),
                                        html.Button(
                                            children='Add to Portfolio',
                                            id='add-to-portfolio-button',
                                            n_clicks=0,
                                            className='control-button'
                                        ),
                                        html.Br(),

                                        html.Hr(),
                                        html.Div(id='selected-tickers-display', style={'marginTop': '20px'}),
                                        dcc.Store(id='stored-tickers', data=[]),  # To store selected tickers across updates                                        
                                        html.Div('Selected Portfolio', className='fullwidth-app-controls-name'),
                                        dcc.Dropdown(
                                            id='portfolio',
                                            options=[],
                                            value=['AAPL', 'MMM'],
                                            multi=True
                                        ),

                                    ]
                                )
                            ),

                            # Analysis Tab
                            dcc.Tab(
                                label='Analysis',
                                value='graph',
                                children=html.Div(
                                    className='control-tab',
                                    children=[
                                        html.Div('Analysis Type', className='fullwidth-app-controls-name'),
                                        dcc.Dropdown(
                                            id='analysis-type',
                                            options=[
                                                {'label': 'Trend', 'value': 'trend'},
                                                {'label': 'Correlation', 'value': 'correlation'},
                                                {'label': 'Allocation', 'value': 'allocation'}

                                            ],
                                            value='trend',
                                            multi=False
                                        ),
                                        
                                        html.Br(),

                                        html.Div('Time Interval', className='fullwidth-app-controls-name'),
                                        dcc.Dropdown(
                                            id='interval',
                                            options=[
                                                {'label': 'Daily', 'value': '1D'},
                                                {'label': 'Weekly', 'value': '1W'},
                                                {'label': 'Monthly', 'value': '1M'},
                                                {'label': 'Annually', 'value': '1A'}
                                            ],
                                            value='1D'
                                        ),
                                        html.Br(),
                                        
                                        html.Div('Optimization Method', className='fullwidth-app-controls-name'),
                                        html.Div([
                                            dcc.Dropdown(
                                                id='optimization-methods',
                                                options=[
                                                    {'label': 'Equal Weights', 'value': 'equal_weights'},
                                                    {'label': 'Maximum Sharpe Ratio', 'value': 'sharpe'},
                                                    {'label': 'Minimum Volatility', 'value': 'volatility'},
                                                    {'label': 'Omega Ratio', 'value': 'omega_ratio'},
                                                    {'label': 'Equal Risk Contribution (ERC)', 'value': 'erc'}
                                                ],
                                                value= None
                                            ),
                                            dcc.Tooltip(id="method-tooltip"),
                                        ]),
                                        html.Br(),
                                        
                                        html.Div(
                                            [
                                                html.Button(
                                                    'Optimize Portfolio',
                                                    id='optimize-portfolio-button',
                                                    n_clicks=0,
                                                    className='control-button'
                                                )
                                            ],
                                            className='portfolio-optimize-container'
                                        ),
                                        
                                    ]
                                )
                            )
                        ]
                    )
                ], style={"margin-top": "20px"},

            ),
 
            dcc.Loading(
                parent_className='app-loading',
                children=html.Div([
            
                    # Placeholder for removed assets message
                    html.Div(
                        id="removed-assets-message",
  
                        style={
                            'color': '#FF6347',
                            'fontSize': '16px',
                            'margin-bottom': '10px',
                            'text-align': 'center',
                        }
                    ),
            
                    # Graph container
                    html.Div(
                        id="graph-container",
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',
                            'width': '100%',
                            'padding': '20px',
                        }
                    ),
            
                    # Table container
                    html.Div(
                        id="table-container",
                        style={
                            'margin-top': '10px',
                            'margin-left': '30px',
                            'display': 'flex',
                            'justify-content': 'flex-start',
                            'align-items': 'center',
                            'width': '100%',
                        }
                    ),
            
                ]), style={"margin-top": "20px"},
            ),
        ]
    )



def callbacks(app):
    """
    Define the callback functions for the Dash application to handle interactivity.
    
    Args:
        app (Dash): The Dash application instance.
    """
    # Utility function to initialize options from tickers
    def create_options(tickers_df):
        """
        Create dropdown options from the tickers DataFrame.

        Args:
            tickers_df (DataFrame): DataFrame containing ticker symbols and security names.

        Returns:
            list: List of dictionaries with label and value for dropdown options.
        """
        return [{'label': row['Security'], 'value': row['Symbol']} for _, row in tickers_df.iterrows()]

    # Callback 1: Update ticker dropdown based on selected industry
    @app.callback(
        Output('ticker-dropdown', 'options'),
        Input('industry-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_ticker_dropdown(selected_industry):
        """
        Update the options in the ticker dropdown based on the selected industry.
        
        Args:
            selected_industry (str): The selected industry from the dropdown.
        
        Returns:
            list: Filtered dropdown options based on the selected industry.
        """
        if not selected_industry:  # If no industry is selected, return no options
            return []
        filtered_df = sp500_df[sp500_df['GICS Sector'] == selected_industry]
        return create_options(filtered_df)

    # Callback 2: Store selected tickers persistently (clearing if portfolio is empty)
    @app.callback(
        [
            Output('stored-tickers', 'data'),
            Output('optimize-portfolio-button', 'n_clicks')  # Reset n_clicks to zero when clearing portfolio
        ],
        Input('add-to-portfolio-button', 'n_clicks'),
        State('ticker-dropdown', 'value'),
        State('stored-tickers', 'data'),
        State('portfolio', 'value'),  # Check the current state of the portfolio
        prevent_initial_call=True
    )
    def store_selected_tickers(store_clicks, selected_tickers, stored_tickers, portfolio):
        """
        Store or update the selected tickers persistently.

        Args:
            store_clicks (int): Number of times the store button has been clicked.
            selected_tickers (list): List of selected tickers from the dropdown.
            stored_tickers (list): List of tickers already stored.
            portfolio (list): Current portfolio state.

        Returns:
            tuple: Updated stored tickers and reset state for the optimize button.
        """
        stored_tickers = stored_tickers or [] 
    
        # If no tickers are selected, clear the portfolio and reset n_clicks
        if not selected_tickers:
            # Reset stored_tickers and optimize portfolio button's n_clicks to 0
            return [], 0  # Clear portfolio and reset n_clicks to 0
    
        # If portfolio is empty, reset the stored tickers and reset n_clicks
        if not portfolio:
            updated_tickers = list(set(selected_tickers))  # Reset to selected tickers
            return updated_tickers, 0  # Reset optimize portfolio button n_clicks to 0
        
        # If there are tickers in the portfolio, update stored tickers by adding the selected ones
        updated_tickers = list(set(stored_tickers + selected_tickers))  # Avoid duplicates
        return updated_tickers, store_clicks  # Keep optimize portfolio button n_clicks unchanged


    # Callback 3: Display stored tickers in portfolio dropdown
    @app.callback(
        [Output('portfolio', 'options'),
         Output('portfolio', 'value')],
        Input('stored-tickers', 'data'),
        prevent_initial_call=True
    )
    def update_portfolio_dropdown(stored_tickers):
        """
        Update the portfolio dropdown options and preselect stored tickers.

        Args:
            stored_tickers (list): List of stored tickers.

        Returns:
            tuple: Dropdown options and preselected values for the portfolio.
        """
        if not stored_tickers:  # If no tickers are stored, return empty options and value
            return [], []
        # Filter stored tickers from the original dataframe to get Security Name
        filtered_df = sp500_df[sp500_df['Symbol'].isin(stored_tickers)]
        options = create_options(filtered_df)
        return options, stored_tickers  # Preselect all stored tickers


    # Callback 4: Update content (graphs, tables, messages) based on portfolio and analysis type
    @app.callback(
        [
            Output("removed-assets-message", "children"),
            Output("graph-container", "children"),
            Output("table-container", "children")
        ],
        [
            Input('portfolio', 'value'),
            Input("analysis-type", "value"),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
            Input("interval", "value"),
            Input('optimize-portfolio-button', 'n_clicks'),
            Input('optimization-methods', 'value')
        ],
        [State("cached-content", "data")],
        prevent_initial_call=True
    )


    def update_content(selected_tickers, analysis_type, start_date, end_date, interval, optimize_clicks, optimization_methods, cached_data):
        """
        Update graphs, tables, and messages based on user interactions.

        Args:
            selected_tickers (list): List of selected tickers.
            analysis_type (str): Selected analysis type (e.g., "trend", "correlation").
            start_date (str): Start date for analysis.
            end_date (str): End date for analysis.
            interval (str): Data interval (e.g., daily, weekly).
            optimize_clicks (int): Number of times optimization is triggered.
            optimization_methods (str): Selected optimization method.
            cached_data (dict): Previously cached data.

        Returns:
            tuple: Updated message, graphs, and tables for display.
        """
        # Handle cases where no tickers are selected
        if not selected_tickers:
            if cached_data:  # Use cached data if available
                return (
                    cached_data.get("removed_message", ""),
                    cached_data.get("graphs", None),
                    cached_data.get("table", None)
                )
            return "", None, None  # Default empty values
    
        results = process_data(
            selected_tickers, start_date, end_date, interval, optimize_clicks, optimization_methods
        )
    
        # Access results
        resampled_data = results["resampled_data"]
        resampled_sp500_data = results["resampled_sp500_data"]
        returns = results["returns"]
        sp500_returns = results["sp500_returns"]
        weights = results["weights"]
        removed_assets = results["removed_assets"]
        weighted_returns = results["metrics"]["weighted_returns"]
        mean_return = results["metrics"]["mean_return"]
        volatility = results["metrics"]["volatility"]
        sharpe_ratio = results["metrics"]["sharpe_ratio"]
    
        removed_message = ""
        
        # Filter out assets with zero or near-zero weights
        filtered_tickers = [ticker for ticker, weight in zip(selected_tickers, weights) if weight > 0.00001]
        filtered_weights = [weight for weight in weights if weight > 0.00001]
    
        # Correlation Table
        returns['^GSPC'] = sp500_returns
    
        filtered_returns = returns[filtered_tickers]
        filtered_returns['^GSPC'] = sp500_returns
    
        portfolio_returns = pd.Series(filtered_returns.mean(axis=1), name='Portfolio')
    
        returns_with_portfolio = filtered_returns.join(portfolio_returns)
    
        # Compute the correlation matrix
        correlation_matrix = returns_with_portfolio.corr()
    
        # Rename the correlation matrix
        correlation_matrix = correlation_matrix.rename(columns={'^GSPC': 'S&P 500'}, index={'^GSPC': 'S&P 500'})
        correlation_matrix = correlation_matrix.rename(columns={'Portfolio': 'Portfolio'}, index={'Portfolio': 'Portfolio'})
    
        # Convert correlation matrix to a list of dictionaries for Dash compatibility
        correlation_data = correlation_matrix.round(2).reset_index()
        correlation_data = correlation_data.rename(columns={'index': 'Ticker'})
        correlation_data_dict = correlation_data.to_dict('records')
    
        # Convert to an HTML table
        correlation_table = dash_table.DataTable(
            data=correlation_data_dict,
            columns=[{"name": col, "id": col} for col in correlation_data.columns]
        )
        # Prepare the message for removed assets
        if removed_assets:
            removed_message = html.Div(
                f"The following assets have been removed due to negligible weights: {', '.join(removed_assets)}",
            className='removed-message',
            )   
    
        # Define theme for tables based on user selection
        table_theme = {
            'header_bgcolor': '#323232',
            'header_font_color': '#D1D3D4',
            'cell_bgcolor': '#232323',
            'cell_font_color': '#D1D3D4',
            'cell_border_color': '1px solid #444',
            'paper_bgcolor': '#1E1E1E',
        }
        
        if analysis_type == "trend":
            # First Graph: Stock Prices (Price Chart)
            price_chart = dcc.Graph(
                id='price-chart',
                figure={
                    'data': [
                        dict(type='scatter', x=resampled_data.index, y=resampled_data[ticker], mode='lines', name=ticker)
                        for ticker in filtered_tickers
                    ],
                    'layout': {
                        'title': {
                            'text': 'Stock Price',
                            'x': 0.5,  # Center align the title
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'pad': {'t': 0} 
                        },
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': 'Price (USD)'},
                        'plot_bgcolor': '#323232',
                        'paper_bgcolor': '#232323',
                        'font': {
                            'color': '#C4C6C8'
                        },
                        'showlegend': True  # Ensure legend is visible in the layout

                    }
                },
                style={
                    'width': '100%',  # Make the width 100% of the parent container
                }
            )

            # Assuming 'selected_tickers' is a list of tickers in the portfolio
            equally_weighted_returns = resampled_data[selected_tickers].pct_change().mean(axis=1).dropna()
            
            # S&P 500 returns for comparison
            sp500_returns = resampled_sp500_data.pct_change().dropna()['^GSPC']
            
            # Return chart including the equally weighted portfolio
            return_chart = dcc.Graph(
                id='return-chart',
                figure={
                    'data': [
                        # Equally Weighted Portfolio
                        dict(type='scatter', x=equally_weighted_returns.index, y=equally_weighted_returns, mode='lines', name='Equally Weighted Portfolio'),
                        
                        # Weighted Portfolio
                        dict(type='scatter', x=weighted_returns.index, y=weighted_returns, mode='lines', name='Optimized Portfolio'),
                        
                        # S&P 500
                        dict(type='scatter', x=sp500_returns.index, y=sp500_returns, mode='lines', name='S&P 500 (^GSPC)')
                    ],
                    'layout': {
                        'title': {
                            'text': 'Stock Returns',
                            'x': 0.5,  # Center align the title
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'pad': {'t': 0}
                        },
                        'xaxis': {'title': 'Date'},
                        'yaxis': {
                            'title': 'Returns (%)',
                            'tickformat': '.1%'  # Format y-axis values as percentages with 2 decimals
                        },
                        'plot_bgcolor': '#323232',
                        'paper_bgcolor': '#232323',
                        'font': {
                            'color': '#C4C6C8'
                        }
                    }
                },
                style={
                    'width': '100%',  # Make the width 100% of the parent container
                }
            )

            # Add security name to metrics table
            metrics_data = []
            for ticker in [*filtered_tickers, '^GSPC']:  # Include selected tickers and benchmark
                # Get security name (e.g., "Apple Inc. (AAPL)")
                security_name = sp500_df.loc[sp500_df['Symbol'] == ticker, 'Security'].values[0] if ticker in sp500_df['Symbol'].values else ticker
                
                ticker_returns = returns[ticker]
                mean_return_ticker = ticker_returns.mean() * 100
                volatility_ticker = ticker_returns.std() * 100
                sharpe_ratio_ticker = ticker_returns.mean() / ticker_returns.std() if ticker_returns.std() != 0 else 0
                weight = weights[selected_tickers.index(ticker)] * 100 if ticker in selected_tickers else None

                metrics_data.append({
                    'Ticker': security_name,
                    # 'Security': security_name,  # Add security name
                    'Mean Return (%)': mean_return_ticker,
                    'Volatility (%)': volatility_ticker,
                    'Sharpe Ratio': sharpe_ratio_ticker,
                    'Weight (%)': weight

                })

            
            portfolio_metrics = {
                'Ticker': 'Portfolio',
                'Mean Return (%)': mean_return * 100,
                'Volatility (%)': volatility * 100,
                'Sharpe Ratio': sharpe_ratio,
                # 'Weights': None #', '.join([f"{w:.2f}" for w in weights])
            }
            metrics_data.append(portfolio_metrics)
            metrics_df = pd.DataFrame(metrics_data)

                
            # Create the metrics table
            metrics_table = dash_table.DataTable(
                columns=[{'name': col, 'id': col} for col in metrics_df.columns],
                data=metrics_df.round(2).to_dict('records'),
                style_table={'margin': '0 auto', 'width': '80%'},
                style_header={
                    'backgroundColor': table_theme['header_bgcolor'],
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'color': table_theme['header_font_color'],  # Dynamically set header font color
                    'padding': '10px',
                },
                style_cell={
                    'backgroundColor': table_theme['cell_bgcolor'],  # Dynamically set cell background color
                    'color': table_theme['cell_font_color'],         # Dynamically set cell font color
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'fontSize': '12px',
                    'border': table_theme['cell_border_color'],  # Light gray borders for cells
        
                }
            )
        
            graphs = [price_chart, return_chart]
            table = [metrics_table]
        


        elif analysis_type == "correlation":
            # Heatmap graph
            heatmap_graph = dcc.Graph(
                id='correlation-heatmap',
                figure={
                    'data': [
                        dict(
                            type='heatmap',
                            z=correlation_matrix.values,                # Use correlation matrix values
                            x=correlation_matrix.columns.tolist(),      # Column labels
                            y=correlation_matrix.columns.tolist(),      # Row labels
                            colorscale='BrBG',                       # Custom colorscale
                        )
                    ],
                    'layout': {
                        'title': 'Correlation Heatmap',
                        'plot_bgcolor': '#323232',
                        'paper_bgcolor': '#232323',
                        'font': {
                            'color': '#C4C6C8'
                        }
                    }
                }
            )

            
            # Define your table
            correlation_table = dash_table.DataTable(
                columns=[{'name': col, 'id': col} for col in correlation_data.columns],
                data=correlation_data.to_dict('records'),
                style_table={'margin': '0 auto', 'width': '80%'},
                style_header={
                    'backgroundColor': table_theme['header_bgcolor'],
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    
                    'color': table_theme['header_font_color'],  # Dynamically set header font color
                    'padding': '10px',
                },
                style_cell={
                    'backgroundColor': table_theme['cell_bgcolor'],  # Dynamically set cell background color
                    'color': table_theme['cell_font_color'],         # Dynamically set cell font color
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'fontSize': '14px',
                    'border': table_theme['cell_border_color'],  # Light gray borders for cells
        
                }
            )
            graphs = [heatmap_graph]
            table = [correlation_table]
        else:  # "allocation"



            # Create the pie chart with filtered data
            allocation_fig = dcc.Graph(
                id='allocation-chart',
                figure={
                    'data': [
                        dict(
                            type='pie',
                            labels=filtered_tickers,  # Only include tickers with non-zero weights
                            values=filtered_weights,   # Only include corresponding weights
                            hoverinfo='label+percent'
                        )
                    ],
                    'layout': {
                        'title': 'Asset Allocation',
                        'plot_bgcolor': '#323232',
                        'paper_bgcolor': '#232323',
                        'font': {
                            'color': '#C4C6C8'
                        }
                    }
                }
            )

            table = dash_table.DataTable(
                columns=[{'name': "Asset", 'id': "Asset"}, {'name': "Share (%)", 'id': "Allocation"}],
                data=[{'Asset': asset, 'Allocation': f"{allocation * 100:.2f}%"} for asset, allocation in zip(filtered_tickers, filtered_weights)],
                style_table={'margin_left': '170px', 'width': '100%'},
                style_header={
                    'backgroundColor': table_theme['header_bgcolor'],
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'color': table_theme['header_font_color'],  # Dynamically set header font color
                    'padding': '10px',
                },
                style_cell={
                    'backgroundColor': table_theme['cell_bgcolor'],  # Dynamically set cell background color
                    'color': table_theme['cell_font_color'],         # Dynamically set cell font color
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'fontSize': '14px',
                    'border': table_theme['cell_border_color'],  # Light gray borders for cells
                },
                style_data_conditional=[
                    {
                        'if': {
                            'column_id': 'Allocation',  # Target the Allocation column
                        },
                        'textAlign': 'center',  # Ensure alignment for consistency
                    }
                ]
            )

                
            graphs = [allocation_fig]

        return removed_message, html.Div(graphs), html.Div(table)

# Run the app
app = run_standalone_app(layout, callbacks, header_colors)
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
