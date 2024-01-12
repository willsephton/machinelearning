### Required Libraries

This Streamlit application requires the following Python libraries to be installed:

- **pandas**: Used for data manipulation and analysis.
- **yfinance**: Enables fetching financial data from Yahoo Finance.
- **scikit-learn (sklearn)**:
  - **preprocessing**: Provides data preprocessing techniques.
  - **decomposition**: Implements PCA (Principal Component Analysis).
  - **cluster**: Implements KMeans clustering algorithm.
- **matplotlib**: Used for creating visualizations and plots.
- **prophet**: Implements Facebook's Prophet for time series forecasting.
- **numpy**: Provides support for numerical operations and arrays.
- **seaborn**: Offers enhanced visualizations based on matplotlib.
- **tensorflow**: Used for machine learning models, especially for neural networks.
- **pickle**: Enables serialization and deserialization of Python objects.
- **statsmodels**: Provides classes and functions for statistical models.
- **pmdarima**: Offers Auto-ARIMA functionality for time series forecasting.
- **StreamLit**: Allows for an easily deployable Python Web Application


To install these libraries, you can use `pip`. For example:

```bash
pip install pandas yfinance scikit-learn matplotlib prophet numpy seaborn tensorflow statsmodels pmdarima streamlit

```

For this project the following stocks were chosen:
- Broadcom Inc. - AVGO
-	Meta Platforms Inc. - META
-	Booking Holdings Inc. - BKNG
-	Tesla Inc. - TSLA

To run this project after installing all the required libraries, open a command terminal in the main folder, or open the project in your IDE and use the terminal in there and type:

```bash
streamlit run main.py

```
