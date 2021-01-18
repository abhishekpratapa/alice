from alice.config import Config
from alice.model import TemporalModel
from alice.data import StockDataLoader
from alice.labeler import BuySellLabeler
from alice.trainer import Trainer

def train_trainer():
    config = Config()
    labeler = BuySellLabeler(
        config.timestep,
        config.num_bins,
        config.step_size
    )

    with open("./tests/config.json") as json_data_file:
        data = json.load(json_data_file)
        alpaca_key = data['alpaca']['key']
        alpaca_secrets = data['alpaca']['secret']

        labeler = BuySellLabeler(
            config.timestep,
            config.num_bins,
            config.step_size
        )

        loader = StockDataLoader(
            'AAPL',
            config.batch_size,
            config.start_date,
            config.end_date,
            alpaca_key,
            alpaca_secrets,
            labeler
        )

        model = TemporalModel(
            config.batch_size,
            config.num_features,
            config.num_classes,
            config.prediction_window,
            config.learning_rate,
            config.conv_h,
            config.lstm_h,
            config.dense_h,
            config.seed
        )

        trainer = Trainer(model,
            loader,
            config.max_iterations
        )
        trainer.train()