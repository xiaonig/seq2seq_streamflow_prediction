import HydroErr as he
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_process import gen_scale_X, gen_scale_Y

import seq2seq

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # check if GPU is available


class CamelsDataset(Dataset):
    """Base data set class to load and preprocess data (batch-first) using PyTorch's Dataset"""
    def __init__(
            self,
            basins: list,
            date_list: list,
            data_attr: pd.DataFrame,
            data_level: pd.DataFrame,
            loader_type: str = "train",
            seq_length: int = 100,
    ):
        super(CamelsDataset, self).__init__()
        self.data_attr = data_attr
        if loader_type not in ["train", "valid", "test"]:
            raise ValueError(
                " 'loader_type' must be one of 'train', 'valid' or 'test' "
            )
        else:
            self.loader_type = loader_type
            self.basins = basins
            self.dates = pd.date_range(date_list[0], date_list[1], freq='H')
            self.seq_length = seq_length
            # data_attr: 前72h的归一化后水位数据
            self.data_attr = data_attr
            # data_level: 后72h的归一化后水位数据
            self.data_level = data_level
            self._load_data()

    def __len__(self):
        return self.num_samples

    def _load_data(self):
        self._create_lookup_table()
        self.c = self.data_attr
        self.y = self.data_level

    def __getitem__(self, item: int):
        # c代表输入数据(前72h的归一化后水位数据)，y代表输出数据(后72h的归一化后水位数据)
        c = self.c.transpose().values[item]
        y = self.y.transpose().values[item]
        c_tensor = torch.unsqueeze(torch.from_numpy(c).float().cuda(), -1)
        y_tensor = torch.unsqueeze(torch.from_numpy(y).float().cuda(), -1)
        return c_tensor, y_tensor

    def _create_lookup_table(self):
        """create an index table for __getitem__ functions"""
        lookup = []
        # list to collect basins ids of basins without a single training sample
        seq_length = self.seq_length
        dates = self.dates.values
        time_length = len(dates)
        for basin in tqdm(self.basins):
            for j in range(time_length - seq_length + 1):
                lookup.append((basin, dates[j]))
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)


def train_epoch(model, optimizer, loader, loss_func, epoch, writer: SummaryWriter):
    """Train model for a single epoch"""
    # set model to train mode (important for dropout)
    model.to(DEVICE)
    model.train()
    pbar = tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        # get model predictions
        y_hat = model(xs, ys)
        ys = ys.squeeze(-1)
        # calculate loss
        loss = loss_func(y_hat, ys)
        writer.add_scalar("Loss/train", loss, epoch)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(str(loss))


def eval_model(model, loader):
    """Evaluate the model"""
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs, ys)
            obs.append(ys)
            preds.append(y_hat)
    return torch.cat(obs), torch.cat(preds)


def set_random_seed(seed):
    print("Random seed:", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_owyy_model():
    set_random_seed(1234)
    # 选择一些站点stcd号进行训练
    chosen_basins = ['10701600', '10412314']
    # 训练集时间范围
    train_times = ['2020-6-1 00:00:00', '2021-10-31 00:00:00']
    # 测试集时间范围
    valid_times = ['2022-6-1 00:00:00', '2022-8-10 01:00:00']
    hidden_size = 4  # Number of LSTM cells
    dropout_rate = 0.0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
    learning_rate = 1e-3  # Learning rate used to update the weights
    sequence_length = 72  # Length of the meteorological record provided to the network
    train_batch_size = 32
    valid_batch_size = 32
    for basin_stcd in chosen_basins:
        train_attrs, train_scaler = gen_scale_X(pd.date_range(train_times[0], train_times[1], freq='H'), stcd=basin_stcd,
                                                pre_index=sequence_length)
        valid_attrs, valid_scaler = gen_scale_X(pd.date_range(valid_times[0], valid_times[1], freq='H'), stcd=basin_stcd,
                                                pre_index=sequence_length)
        train_level_attrs, train_level_scaler = gen_scale_Y(pd.date_range(train_times[0], train_times[1], freq='H'), stcd=basin_stcd,
                                                            post_index=sequence_length)
        valid_level_attrs, valid_level_scaler = gen_scale_Y(pd.date_range(valid_times[0], valid_times[1], freq='H'), stcd=basin_stcd,
                                                            post_index=sequence_length)
        # Training data
        ds_train = CamelsDataset(
            basins=[basin_stcd],
            date_list=train_times,
            data_attr=train_attrs,
            data_level=train_level_attrs,
            loader_type="train",
            seq_length=sequence_length,
        )
        tr_loader = DataLoader(ds_train, batch_size=train_batch_size, shuffle=True)
        ds_val = CamelsDataset(
            basins=[basin_stcd],
            date_list=valid_times,
            data_attr=valid_attrs,
            data_level=valid_level_attrs,
            loader_type="valid",
            seq_length=sequence_length,
        )
        val_loader = DataLoader(ds_val, batch_size=valid_batch_size, shuffle=False)
        writer = SummaryWriter()
        # Here we create our model, feel free
        model = seq2seq.Seq2Seq(encoder=seq2seq.Encoder(1, hidden_size, 1, batch_first=True, dropout=dropout_rate),
                                decoder=seq2seq.Decoder(1, hidden_size, 1, batch_first=True, dropout=dropout_rate),
                                device=DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_func = nn.MSELoss()
        n_epochs = 15  # Number of training epochs
        print('Outputting ' + basin_stcd + ' results')
        for i in range(n_epochs):
            train_epoch(model, optimizer, tr_loader, loss_func, i + 1, writer)
            obs, preds = eval_model(model, val_loader)
            preds_df = pd.DataFrame(preds.cpu().numpy())
            obs_df = pd.DataFrame(obs.squeeze(-1).cpu().numpy())
            preds = valid_level_scaler.inverse_transform(preds_df)
            obs = valid_level_scaler.inverse_transform(obs_df)
            # obs = obs.cpu().numpy().reshape(basins_num, -1)
            # preds = preds.values.reshape(basins_num, -1)
            nse = np.array([he.nse(preds[i], obs[i]) for i in range(obs.shape[0])])
            tqdm.write(f"Validation NSE mean: {nse.mean():.2f}")
        print('_________________________________________')
