hparams = {
    "epochs": 250,
    "learning_rate": 1e-4,
    "num_features": 40,
    "out_vocab": 35, # Including Blanks
    "weight_decay": 5e-5,
    "batch_size": 128,
}

def get_hyperparameters():
    return hparams
