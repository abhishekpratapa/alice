from alice.config import Config
from alice.model import TemporalModel

def test_model_init():
    config = Config()
    model = TemporalModel(
        config.batch_size,
        config.num_features,
        config.model_config['num_classes'],
        config.prediction_window,
        config.learning_rate,
        config.model_config['conv_h'],
        config.model_config['lstm_h'],
        config.model_config['dense_h'],
        config.seed
    )
