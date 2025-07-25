 Stock Price Prediction with LSTM Neural Networks

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical data from Yahoo Finance.

 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)
- [License](#license)

 Overview
This Jupyter notebook demonstrates:
1. Data collection of AAPL stock prices (2010-2023)
2. Data preprocessing and feature engineering
3. Building and training an LSTM model
4. Evaluating model performance
5. Visualizing predictions vs actual prices

 Features
- Automated data fetching using `yfinance`
- MinMax scaling for data normalization
- Sequence generation for time-series prediction
- Customizable LSTM architecture
- Performance metrics (MSE, RMSE, MAE)
- Interactive visualizations
  
 Installing Dependencies
  pip install -r requirements.txt
