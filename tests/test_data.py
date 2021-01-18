from alice.config import Config
from alice.data import StockDataLoader
from alice.labeler import BuySellLabeler

import json

def test_data():
    config = Config()
    with open("./tests/config.json") as json_data_file:
        data = json.load(json_data_file)
        alpaca_key = data['alpaca']['key']
        alpaca_secrets = data['alpaca']['secret']
        labeler = BuySellLabeler(
            config.timestep,
            config.num_bins,
            config.step_size
        )

        data_loader = StockDataLoader(
            'AAPL',
            config.batch_size,
            config.start_date,
            config.end_date,
            alpaca_key,
            alpaca_secrets,
            labeler
        )
