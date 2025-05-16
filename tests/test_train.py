from src.train import train, WSIModel

def test_train():
    """Test that the training pipeline executes without crashing"""

    train(
        "data/",
        'labels.csv',
        WSIModel(),
        max_epochs=1
    )