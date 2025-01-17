# Portfolio Selection Dash Application

This is a Dash web application that allows users to select assets, optimize their portfolio, and analyze various performance metrics. It provides interactive tools for selecting tickers, performing portfolio optimization, and analyzing performance through various charts and tables.

## Features

- **Industry-Based Ticker Selection**: Users can filter tickers based on the industry (sector) they are interested in.
- **Portfolio Construction**: Users can select tickers from a filtered list and add them to a portfolio for further analysis.
- **Portfolio Optimization**: The app supports several optimization methods including:
  - **Equally Weighted Portfolio**
  - **Mean-Variance Optimization (MVO)**
  - **Maximum Sharpe Ratio Portfolio**
  - **Hierarchical Risk Parity (HRP)**
- **Performance Analysis**:
  - **Trend Analysis**: Plot stock prices and returns over time for the selected tickers.
  - **Correlation Analysis**: Display a correlation matrix between the selected tickers and the S\&P 500.
  - **Asset Allocation**: Visualize asset allocation using pie charts and tables.

## Installation

1. Clone the repository or download the code.
2. Install the required dependencies by running the following command:
   
   ```bash
   pip install -r requirements.txt
