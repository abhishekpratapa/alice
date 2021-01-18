from alice.config import Config

def test_config_init():
    config = Config()
    assert config.batch_size == 32
    assert config.timestep == 60
    assert config.multiple == 13
    assert config.num_bins == 64
    assert config.rolling_window == 12
    assert config.num_features == 60
    assert config.step_size == 1.0
    assert config.prediction_window == 130
    assert config.seed == 1248
    assert config.max_iterations == 2000
    assert config.learning_rate == 1e-3
    assert config.model_config['conv_h'] == 64
    assert config.model_config['lstm_h'] == 200
    assert config.model_config['dense_h'] == 128
    assert config.model_config['num_classes'] == 2