import sys
sys.path.insert(1, '../../src/')

import alice

if __name__ == '__main__':
    config = alice.config.load_config('./config.json')

    labeler = alice.labeler.SellLabeler(
        config.timestep,
        config.num_bins,
        config.step_size
    )

    loader = alice.data.StockDataLoader(
        'AAPL',
        config.batch_size,
        config.start_date,
        config.end_date,
        config.min_sequence,
        config.max_sequence,
        config.rolling_window,
        config.num_bins,
        config.alpaca_key,
        config.alpaca_secrets,
        labeler
    )

    model = alice.model.TemporalModel(
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

    trainer = alice.trainer.Trainer(model,
        loader,
        config.max_iterations
    )

    trainer.train()
