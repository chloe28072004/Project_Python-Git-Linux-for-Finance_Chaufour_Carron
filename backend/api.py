"""
Flask API Backend
Exposes endpoints for Quant A and Quant B modules
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from quant_a import QuantA
from quant_b import QuantB
import traceback

app = Flask(__name__)
CORS(app)  


@app.route('/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({"status": "ok", "message": "Finance Dashboard API is running"})


# Quant A endpoints

@app.route('/api/quant-a/price/<symbol>', methods=['GET'])
def get_current_price(symbol):
    """Get current price for a single asset"""
    try:
        quant = QuantA(symbol)
        quant.update_current_price()
        return jsonify({
            "symbol": symbol,
            "current_price": quant.current_price,
            "timestamp": quant.get_timestamp() if hasattr(quant, 'get_timestamp') else None
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/quant-a/historical/<symbol>', methods=['GET'])
def get_historical_data(symbol):
    """Get historical data for a single asset"""
    try:
        days = request.args.get('days', default=30, type=int)
        quant = QuantA(symbol)
        data = quant.get_historical_data(days=days)

        # converting dataframe to JSON if needed
        if data is not None:
            data_json = data.to_dict('records') if hasattr(data, 'to_dict') else data
        else:
            data_json = []

        return jsonify({
            "symbol": symbol,
            "data": data_json
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/quant-a/strategy/<symbol>', methods=['POST'])
def run_strategy(symbol):
    """Run a strategy on a single asset"""
    try:
        data = request.get_json()
        strategy_type = data.get('strategy', 'buy_and_hold')
        initial_capital = data.get('initial_capital', 10000)

        quant = QuantA(symbol)
        quant.get_historical_data()

        if strategy_type == 'buy_and_hold':
            result = quant.strategy_buy_and_hold(initial_capital=initial_capital)
        elif strategy_type == 'momentum':
            window = data.get('window', 20)
            result = quant.strategy_momentum(window=window, initial_capital=initial_capital)
        else:
            return jsonify({"error": "Unknown strategy"}), 400

        return jsonify({
            "symbol": symbol,
            "strategy": strategy_type,
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/quant-a/metrics/<symbol>', methods=['GET'])
def get_metrics(symbol):
    """Get performance metrics for a single asset"""
    try:
        quant = QuantA(symbol)
        metrics = quant.get_performance_metrics()
        return jsonify({
            "symbol": symbol,
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# Quant B endpoints

@app.route('/api/quant-b/prices', methods=['POST'])
def get_portfolio_prices():
    """Get current prices for multiple assets"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])

        if not symbols:
            return jsonify({"error": "No symbols provided"}), 400

        quant = QuantB(symbols)
        quant.update_current_prices()

        return jsonify({
            "symbols": symbols,
            "prices": quant.data
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/quant-b/portfolio', methods=['POST'])
def run_portfolio_strategy():
    """Run portfolio strategy on multiple assets"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        strategy_type = data.get('strategy', 'equal_weight')
        initial_capital = data.get('initial_capital', 10000)
        weights = data.get('weights', None)

        if not symbols:
            return jsonify({"error": "No symbols provided"}), 400

        quant = QuantB(symbols)
        quant.get_historical_data()

        if strategy_type == 'equal_weight':
            result = quant.equal_weight_portfolio(initial_capital=initial_capital)
        elif strategy_type == 'custom_weight':
            if not weights:
                return jsonify({"error": "Weights required for custom_weight strategy"}), 400
            result = quant.custom_weight_portfolio(weights=weights, initial_capital=initial_capital)
        else:
            return jsonify({"error": "Unknown strategy"}), 400

        return jsonify({
            "symbols": symbols,
            "strategy": strategy_type,
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/quant-b/metrics', methods=['POST'])
def get_portfolio_metrics():
    """Get portfolio metrics"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])

        if not symbols:
            return jsonify({"error": "No symbols provided"}), 400

        quant = QuantB(symbols)
        quant.get_historical_data()
        metrics = quant.get_portfolio_metrics()

        # converting correlation matrix to list if it's a numpy array
        if metrics.get('correlation_matrix') is not None:
            if hasattr(metrics['correlation_matrix'], 'tolist'):
                metrics['correlation_matrix'] = metrics['correlation_matrix'].tolist()

        return jsonify({
            "symbols": symbols,
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# run server

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
