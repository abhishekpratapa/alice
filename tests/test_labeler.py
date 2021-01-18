from alice.config import Config
from alice.labeler import BuyLabeler
from alice.labeler import SellLabeler

def test_buy_labeler():
    config = Config()
    labeler = BuyLabeler(
        config.timestep,
        config.num_bins,
        config.step_size
    )

def test_buy_labeler():
    config = Config()
    labeler = SellLabeler(
        config.timestep,
        config.num_bins,
        config.step_size
    )
