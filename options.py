opt = {
    "data_root_path":"./LibriSpeech/train-clean-100/",
    "dev":"cuda:0",
    "dev_list":[0],
    "debug_fraction": 1.0, # set to 1.0 if not debugging
    "lr_step_rate":10,
    "init_learning_rate":.001,
    "lr_step_factor":0.25,
    "batch_size":120,
    "n_epochs":20,
    "num_workers":12,
    "test_dev":"cuda:0",
    "test_batch_size":32,
    "tsne_spk_frac": 0.1,
}