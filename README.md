# Simple-Trading-Bot-
This project demonstrates the development of a machine learning-based trading strategy for SPY (S&P 500 ETF) using Python. The strategy utilizes various technical indicators to predict market movements and generate buy/sell/hold signals. The goal is to evaluate the performance of different machine learning models on real-world financial data and simulate a trading strategy.

Project Overview
The project uses the following technical indicators:

EMA (Exponential Moving Average): Trend-following indicator to identify market direction.
RSI (Relative Strength Index): Momentum indicator to identify overbought or oversold conditions.
Bollinger Bands: Volatility indicator to identify potential price reversals.
MACD (Moving Average Convergence Divergence): A momentum-based indicator to track trend shifts.
OBV (On-Balance Volume): Volume indicator used to confirm trends.
Machine learning models, including Linear Regression, Logistic Regression, Random Forest, and Gradient Boosting, are trained to predict the next day's market movement, generating trading signals (Buy, Sell, Hold). The models are evaluated on their predictive accuracy, precision, recall, and F1-score. A simulated trading strategy is applied to assess model performance with real financial data.

Features
Data Fetching: Fetch historical stock data for SPY (S&P 500 ETF) using Yahoo Finance API.
Technical Indicators: Calculation of various technical indicators including EMA, RSI, Bollinger Bands, MACD, and OBV.
Trading Strategy: Implementation of a trading strategy based on machine learning predictions.
Performance Metrics: Evaluation of model performance using accuracy, precision, recall, F1-score, and the total return of the trading strategy.
Simulated Trading: Backtesting of the trading strategy on historical data to simulate real-world trading performance.
Prerequisites
Python 3.x
Libraries: yfinance, ta, sklearn, numpy, pandas, matplotlib
