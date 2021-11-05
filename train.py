import os
from options import opt
import time
import json
from tqdm import tqdm
import torch
from CPCLibriSpeech.data_management import get_data
from CPCLibriSpeech.model_management import build_models
import gc
import load_mne 

# To run
# $ ./setup.sh
# $ python train.py
# $ python3 test.py ./models/{model_timestamp}
# where {model_timestamp} is the name of the last/most recent folder created during training in models/

dev = opt["dev"]
lr_step_rate = opt["lr_step_rate"]
root_data_path = opt["data_root_path"]
dev_list = opt["dev_list"]
batch_size = opt["batch_size"]
num_workers = opt["num_workers"]
init_learning_rate = opt["init_learning_rate"]
lr_step_factor = opt["lr_step_factor"]
n_epochs = opt["n_epochs"]
reuse_datasplit = opt["reuse_datasplit"]
path_to_best_mode = opt["path_to_best_mode"]


if __name__ == '__main__':

    od = "./models/" + str(int(time.time()*1000))

    os.mkdir(od)

    model = build_models.CPC_LibriSpeech_Encoder()
    
    DP_model = torch.nn.DataParallel(model, dev_list, dev).cuda()

    
    train_s = []
    test_s = []
    train_p = []
    test_p = []
    

    if reuse_datasplit == False:
        (train_p, train_s), (test_p, test_s) = get_data.get_train_test_split(
            root_data_path, test_frac=0.4)

        json.dump(train_s, open(od + "/train_speakers.txt", "w"))
        json.dump(test_s, open(od + "/test_speakers.txt", "w"))
    else: 
        train_s = json.load(open(path_to_best_mode + "/train_speakers.txt"))
        test_s = json.load(open(path_to_best_mode + "/test_speakers.txt"))
        train_p, test_p  =  get_data.load_dataset(train_s, test_s)

    '''
    train_dataset = torch.utils.data.DataLoader(
        train_p, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataset = torch.utils.data.DataLoader(
        test_p, batch_size=batch_size, num_workers=num_workers)

'''
    train_dataset, test_dataset = load_mne.get_dataloader_MNE(batch_size)
    

    print("Train:", len(train_dataset), "samples")
    print("Test:", len(test_dataset), "samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    lr_step = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lambda epoch: lr_step_factor)

    best_loss = None
    for epoch in range(n_epochs):

        print("Epoch " + str(epoch+1))
        torch.cuda.empty_cache()
        for phase in ["train", "test"]:
            print(phase)
            if phase == "train":
                dataset = train_dataset
                DP_model.train()
            if phase == "test":
                dataset = test_dataset
                DP_model.eval()

            running_loss = 0
            for B, spk in tqdm(dataset):
                B.to(dev)

                loss = torch.stack(DP_model(B))
                loss = -torch.mean(loss)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.detach().item()
                del loss
                del B
                

            running_loss /= len(dataset)

            print(phase + "loss: " + str(running_loss))
            if phase == "test" and (best_loss is None or running_loss < best_loss):
                torch.save(model.state_dict(), od + "/best_model_params")

        if (epoch+1) % lr_step_rate == 0:
            lr_step.step()

        torch.cuda.empty_cache()
        

        gc.collect()

        

    torch.save(model.state_dict(), od + "/model_params_final")
