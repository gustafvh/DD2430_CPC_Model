from sklearn.model_selection import ShuffleSplit

from mne import Epochs, pick_types, events_from_annotations, make_fixed_length_events
from mne.io import concatenate_raws, read_raw_edf, read_raw_fif
from mne.datasets import eegbci
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler 



class EpochsDataset(Dataset):
    """Class to expose an MNE Epochs object as PyTorch dataset
    Parameters
    ----------
    epochs_data : 3d array, shape (n_epochs, n_channels, n_times)
        The epochs data.
    epochs_labels : array of int, shape (n_epochs,)
        The epochs labels.
    transform : callable | None
        The function to eventually apply to each epoch
        for preprocessing (e.g. scaling). Defaults to None.
    """
    def __init__(self, epochs_data, epochs_labels, transform=None):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.transform = transform

    def __len__(self):
        return len(self.epochs_labels)

    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X[None, ...])
        return X, y


def get_subject_id(filepath):
    return filepath.split('_')[0].split('-')[-1]


def get_state_id(filepath):
    if 'ses-con_task-rest_ec' in filepath:
        return 1
    if 'ses-con_task-rest_eo' in filepath:
        return 2
    if 'ses-psd_task-rest_ec' in filepath:
        return 3
    if 'ses-psd_task-rest_eo' in filepath:
        return 4
    raise ValueError

    
def fetch_data(subject_ids, state_ids):
    """ fetches all raw.fif MEG files
    and returns a list of triplets. each triplet
    containing (subject_id, state_id, filepath).
    """
    subject_ids = list(map(lambda x: '0'+str(x) if len(str(x)) != 2 else str(x), subject_ids))
    stateid_map = {1: 'ses-con_task-rest_ec',
                   2: 'ses-con_task-rest_eo',
                   3: 'ses-psd_task-rest_ec',
                   4: 'ses-psd_task-rest_eo'}
    
    homedir = os.path.expanduser('~')
    files = list(os.listdir(os.path.join('data/')))
    files = list(os.path.join('data/'+f) for f in files if get_subject_id(f) in subject_ids)
    
    subject_state_files = list()
    for file in files:
        for state in state_ids:
            if stateid_map[state] in file:
                subject_state_files.append(file)
    
    subject_state_files = list((get_subject_id(f), get_state_id(f), f) for f in subject_state_files)
    return subject_state_files

def get_data():
    tmin, tmax = -1., 4.
    event_id = dict()
    subject = 1
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    subject_ids = list(range(1, 2))
    state_ids = list(range(1, 5))
    file_path = fetch_data(subject_ids, state_ids)


    raw = concatenate_raws([read_raw_fif(f[2], preload=True) for f in file_path])

    exclude = ['IASX+', 'IASX-', 'IASY+', 'IASY-', 'IASZ+', 'IASZ-', 'IAS_DX', 'IAS_X', 'IAS_Y', 'IAS_Z',
                   'SYS201','CHPI001','CHPI002','CHPI003', 'CHPI004','CHPI005','CHPI006','CHPI007','CHPI008',
                   'CHPI009', 'IAS_DY', 'MISC001','MISC002', 'MISC003','MISC004','MISC005', 'MISC006']
    
    raw.drop_channels(exclude)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    #events, _ = events_from_annotations(raw)

    events = make_fixed_length_events(raw, start=5, stop=50, duration=2.)

    picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, tmin=0.0, tmax=50, baseline=(0, 0))
    
    labels = epochs.events[:, 2] - 2
    return epochs.get_data()[:, :, :], labels

def get_dataloader_MNE(batch_size):

    epochs_data, labels = get_data()

    dataset = EpochsDataset(epochs_data, labels)


    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data)
    train_idx, test_idx = next(cv_split)

    ds_train, ds_valid = Subset(dataset, train_idx), Subset(dataset, test_idx)
    

    # batch_size_train = len(ds_train)
    # batch_size_valid = len(ds_valid)


    sampler_train = RandomSampler(ds_train)
    sampler_valid = SequentialSampler(ds_valid)

    # create loaders
    num_workers = 1
    loader_train = \
        DataLoader(ds_train, batch_size=2040,
                num_workers=num_workers, sampler=sampler_train)
    loader_valid = \
        DataLoader(ds_valid, batch_size=2040,
                num_workers=num_workers, sampler=sampler_valid)

    print("loader train:", len(loader_train), " loader_valid: ", len(loader_valid))

    return loader_train, loader_valid

t, s = get_dataloader_MNE(120)