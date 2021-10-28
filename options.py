opt = {
    "data_root_path": "./LibriSpeech/train-clean-100/",
    "dev": "cpu",
    "dev_list": [0, 2],
    "lr_step_rate": 10,
    "init_learning_rate": .001,
    "lr_step_factor": 0.25,
    "batch_size": 8,
    "n_epochs": 2,
    "num_workers": 24,
    "test_dev": "cpu",
    "test_batch_size": 64,
    "tsne_spk_frac": .1,
}
