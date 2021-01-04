from flask import escape, jsonify
import firebase_admin
import pandas as pd
import numpy as np
from datetime import datetime
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


def getPortfolioFigures(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.

    """

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'btc_allocation' in request_json and 'eth_allocation' in request_json and 'gold_allocation' in request_json:
        btc_allocation = request_json['btc_allocation']
        eth_allocation = request_json['eth_allocation']
        gold_allocation = request_json['gold_allocation']
        past_years = request_json['past_years']

        # calculate start time
        end = datetime.today()
        start = datetime(end.year-int(past_years), end.month, end.day)

        # read prices from firestore and put into df
        prices_fs = list(db.collection(u'prices').where(
            u'timestamp', u'>=', start).stream())
        prices_dict = list(map(lambda x: x.to_dict(), prices_fs))
        prices = pd.DataFrame(prices_dict)
        prices.sort_values(by=['timestamp'], inplace=True)
        prices.set_index('timestamp', inplace=True)

        returns = prices.pct_change()

        # Construct a covariance matrix for the daily returns data
        cov_matrix_d = returns.cov()

        # Annualize matrix
        cov_matrix_a = (cov_matrix_d) * 365

        # allcoation of different assets
        weights = np.empty(3)
        weights[prices.columns.get_loc("btc")] = float(btc_allocation)
        weights[prices.columns.get_loc("eth")] = float(
            eth_allocation)
        weights[prices.columns.get_loc("gold")] = float(gold_allocation)

        # Calculate daily portfolio returns
        returns['Portfolio'] = returns.dot(weights)

        # Compound the percentage returns over time
        daily_cum_ret = (1+returns).cumprod()

        # Portfolio standard deviation
        port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_a, weights)))

        # Total return of portfolio
        total_return = daily_cum_ret.Portfolio[-1] - 1

        # Calculate annualized return over 3 years
        annualized_return = ((1 + total_return)**(1/int(past_years)))-1

        # Calcuate the sharpe ratio
        sharpe_ratio = (annualized_return - 0) / port_stddev

        # Calculate the maximum value of returns using rolling().max()
        roll_max = daily_cum_ret['Portfolio'].rolling(
            min_periods=1, window=365).max()
        # Calculate daily draw-down from rolling max
        daily_drawdown = daily_cum_ret['Portfolio']/roll_max - 1.0
        # Calculate maximum daily draw-down
        max_daily_drawdown = daily_drawdown.rolling(
            min_periods=1, window=365).min()
        max_drawdown = max_daily_drawdown.min()

        msg = {"total_return": str(total_return), "annualized_return": str(annualized_return), "risk": str(
            port_stddev), "max_drawdown": str(max_drawdown), "sharpe_ration": sharpe_ratio}

    else:
        msg = 'Error'

    return jsonify(msg)
