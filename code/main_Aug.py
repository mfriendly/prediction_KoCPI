import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from data_utils import *
from training_utils import EarlyStopping


class TimeSeriesDataset(Dataset):

    def __init__(self, config, args, data, past_steps, n_features, date_list):
        self.data = data
        self.past_steps = past_steps
        self.n_features = n_features
        self.args = args
        self.date_list = date_list

    def __len__(self):
        return len(self.data) - self.past_steps - self.args.future_steps

    def __getitem__(self, idx):
        X = torch.Tensor(self.data[idx:idx +
                                   self.past_steps, :self.n_features])
        X_sentiment = torch.Tensor(self.data[idx:idx + self.past_steps,
                                             -1]).unsqueeze(1)

        if self.args.sentiment:
            X = torch.cat((X, X_sentiment), dim=1)

        Y = torch.Tensor(self.data[idx + self.past_steps:idx +
                                   self.past_steps + self.args.future_steps,
                                   0:1])

        past_dates = self.date_list[idx:idx + self.past_steps]

        future_dates = self.date_list[idx + self.past_steps:idx +
                                      self.past_steps + self.args.future_steps]

        return X, Y, (past_dates[0], past_dates[-1]), (future_dates[0],
                                                       future_dates[-1])


class Linear(nn.Module):

    def __init__(self, config, args):
        super(Linear, self).__init__()
        self.linear_layer1 = nn.Linear(args.past_steps, args.hidden_dim)
        self.linear_layer2 = nn.Linear(args.hidden_dim, args.future_steps)

    def forward(self, x):
        print(f"==>> x.shape: {x.shape}")
        x = x.permute(0, 2, 1)
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = x.permute(0, 2, 1)
        return x


class LSTM(nn.Module):

    def __init__(self, config, args):
        super(LSTM, self).__init__()
        hidden_dim, step, dropout = args.hidden_dim, args.past_steps, args.dropout
        self.LSTM_layer = nn.LSTM(input_size=args.total_features,
                                  hidden_size=hidden_dim,
                                  batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim, args.future_steps)

    def forward(self, x):

        out, _ = self.LSTM_layer(x)
        out = self.dropout(out[:, -1, :])
        out = self.dense(out)
        return out.unsqueeze(-1)


class CNNLSTM(nn.Module):

    def __init__(self, config, args):
        super(CNNLSTM, self).__init__()
        hidden_dim, step = args.hidden_dim, args.past_steps

        self.CNN1_layer = nn.Conv1d(in_channels=args.total_features,
                                    out_channels=args.hidden_dim,
                                    kernel_size=args.kernel_size)
        self.CNN2_layer = nn.Conv1d(in_channels=args.hidden_dim,
                                    out_channels=args.hidden_dim,
                                    kernel_size=args.kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=args.kernel_size)
        self.LSTM_layer = nn.LSTM(input_size=args.hidden_dim,
                                  hidden_size=hidden_dim,
                                  batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.dense = nn.Linear(hidden_dim, args.future_steps)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.CNN1_layer(x)

        x = self.CNN2_layer(x)
        try:
            cnn2_output = x.detach().cpu().numpy()
            print(f"==>> cnn2_output.shape: {cnn2_output.shape}")

        except:
            pass
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.LSTM_layer(x)
        out = self.dropout(out[:, -1, :])
        out = self.dense(out)
        return out.unsqueeze(-1)


class LinearLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def evaluation(true_vals, pred_vals, future_steps):
    print(f"==>> true_vals.shape: {true_vals.shape}")
    print(f"==>> pred_vals.shape: {pred_vals.shape}")

    mae = np.nanmean(np.abs(true_vals - pred_vals))
    mse = np.nanmean((true_vals - pred_vals)**2)
    rmse = np.sqrt(mse)

    epsilon = 1e-8
    mape = np.nanmean(np.abs(
        (true_vals - pred_vals) / (true_vals + epsilon))) * 100
    smape = SMAPE(true_vals, pred_vals)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "SMAPE": smape,
    }

    return metrics


def load_checkpoint(config, args, model, optimizer, device):
    try:
        checkpoint = torch.load(config["L_PATH"], map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.dtype == "double":
            model = model.to(torch.double)
            print("double")

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"■■■Loaded L_PATH{config['L_PATH']}")
    except Exception as e:
        print(f"Error loading checkpoint from {config['L_PATH']}: {e}")
        try:
            checkpoint = torch.load(config["C_PATH"], map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            if args.dtype == "double":
                model = model.to(torch.double)
                print("double")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = 0
            print(f"■■■Loaded C_PATH{config['C_PATH']}")
        except Exception as e:
            print(f"Error loading checkpoint from {config['C_PATH']}: {e}")
            start_epoch = 0

    config = count_param_and_update_config(config, model)
    return config, start_epoch, model, optimizer


def train_model(config, args, df, model_name):

    start_epoch = 0
    best_val_loss = np.inf
    past_steps = config["past_steps"]
    future_steps = config["future_steps"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    n_features = args.n_features

    f = "../code/lasso_results/col_order.json"
    with open(f) as f:
        cols = json.load(f)

    cols = ["date"] + [args.target_col] + cols
    print("cols", cols)
    df = df[cols]
    df["date"] = pd.to_datetime(df["date"])
    if True:
        train_set, val_set, test_set, trainval_set = slicing_by_date(df)
    train_set_dates = train_set.index.tolist()
    train_set_unix_dates = [timestamp_to_unix(d) for d in train_set_dates]
    train_set_dates_string = [format_date(d) for d in train_set_unix_dates]
    val_set_dates = val_set.index.tolist()
    val_set_unix_dates = [timestamp_to_unix(d) for d in val_set_dates]
    val_set_dates_string = [format_date(d) for d in val_set_unix_dates]
    test_set_dates = test_set.index.tolist()
    test_set_unix_dates = [timestamp_to_unix(d) for d in test_set_dates]
    test_set_dates_string = [format_date(d) for d in test_set_unix_dates]

    train_statistics = calculate_statistics(train_set, cols)
    val_statistics = calculate_statistics(val_set, cols)
    test_statistics = calculate_statistics(test_set, cols)

    train_start, train_end = train_set_dates_string[0], train_set_dates_string[
        -1]
    val_start, val_end = val_set_dates_string[0], val_set_dates_string[-1]
    test_start, test_end = test_set_dates_string[0], test_set_dates_string[-1]
    print(f"Train set start: {train_start}, end: {train_end}")
    print(f"Validation set start: {val_start}, end: {val_end}")
    print(f"Test set start: {test_start}, end: {test_end}")

    train_statistics = train_statistics.round(2)
    val_statistics = val_statistics.round(2)
    test_statistics = test_statistics.round(2)
    train_statistics["dataset"] = "train"
    train_statistics["start_date"] = train_start
    train_statistics["end_date"] = train_end

    val_statistics["dataset"] = "val"
    val_statistics["start_date"] = val_start
    val_statistics["end_date"] = val_end

    test_statistics["dataset"] = "test"
    test_statistics["start_date"] = test_start
    test_statistics["end_date"] = test_end

    combined_statistics = pd.concat(
        [train_statistics, val_statistics, test_statistics], axis=0)[[
            "dataset", "count", "mean", "std", "median", "min", "max",
            "start_date", "end_date"
        ]]
    combined_statistics = combined_statistics[combined_statistics.index ==
                                              "total_index"]
    os.makedirs('z_csv',exist_ok=True)
    if args.augment:
        combined_statistics.to_csv(
            "z_csv/AUG_combined_statistics_vertical.csv", index=True)
    else:
        combined_statistics.to_csv(
            "z_csv/XAUG_combined_statistics_vertical.csv", index=True)

    scaler = MinMaxScaler()

    trainval_set_scaled = scaler.fit_transform(trainval_set)

    scaler_save_path = os.path.join(args.save_path, "scaler_params.json")
    save_scaler_params(scaler, scaler_save_path)
    train_set_scaled = scaler.transform(train_set)
    val_set_scaled = scaler.transform(val_set)
    test_set_scaled = scaler.transform(test_set)
    train_dataset = TimeSeriesDataset(config, args, train_set_scaled,
                                      past_steps, n_features,
                                      train_set_unix_dates)
    val_dataset = TimeSeriesDataset(config, args, val_set_scaled, past_steps,
                                    n_features, val_set_unix_dates)
    test_dataset = TimeSeriesDataset(config, args, test_set_scaled, past_steps,
                                     n_features, test_set_unix_dates)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_name = args.MODEL.split("-F")[0]
    print("model_name", model_name)
    if model_name == "CNNLSTM":
        model = CNNLSTM(config, args)
    elif model_name == "LSTM":
        model = LSTM(config, args)
    elif model_name == "Linear":
        model = Linear(config, args)

    model = model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    config, start_epoch, model, optimizer = load_checkpoint(
        config, args, model, optimizer, args.device)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    if start_epoch != 0:
        try:
            with open(args.save_path + "/train_losses.json", "r") as f:
                train_losses = json.load(f)

            with open(args.save_path + "/val_losses.json", "r") as f:
                val_losses = json.load(f)
        except:
            train_losses = []
            val_losses = []
    else:
        train_losses = []
        val_losses = []
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_idx, (trainX, trainY, past_date,
                        future_date) in enumerate(train_loader):
            trainX, trainY = trainX.to(args.device), trainY.to(args.device)
            trainY = trainY
            optimizer.zero_grad()
            output = model(trainX)
            if False:
                if output.dim == 4:
                    output = output.squeeze(-1)
                if trainY.dim == 4:
                    trainY = trainY.squeeze(-1)

            loss = criterion(output, trainY)

            loss = torch.sqrt(loss)

            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        model.eval()

        with torch.no_grad():
            for valX, valY, past_date, future_date in val_loader:
                valX, valY = valX.to(args.device), valY.to(args.device)
                val_output = model(valX)

                val_loss = criterion(val_output, valY)
                val_losses.append(val_loss.item())

        val_loss_mean = np.mean(val_losses)
        print(
            f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss_mean:.4f}"
        )
        if True:
            if val_loss_mean < best_val_loss and not np.isnan(val_loss_mean):
                best_val_loss = val_loss_mean
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    args.C_PATH,
                )

        early_stopping(val_loss_mean, model, args.save_path)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

        print(
            f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {np.mean(val_losses):.4f}"
        )

    model.eval()
    test_predictions = []
    test_true_vals = []
    test_future_dates = []
    index = 0
    with torch.no_grad():
        for testX, testY, past_date, future_date in test_loader:
            start = format_date(future_date[0].item())

            end = format_date(future_date[1].item())

            if not format_date(future_date[0].item()).endswith("-01"):

                continue
            index += 1
            testX, testY = testX.to(args.device), testY.to(args.device)

            test_output = model(testX)

            first_output = test_output[:, 0, :]
            first_true = testY[:, 0, :]
            test_predictions.append(
                first_output.reshape(1, -1).detach().cpu().numpy())
            test_true_vals.append(
                first_true.reshape(1, -1).detach().cpu().numpy())
            test_future_dates.append(start)

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_true_vals = np.concatenate(test_true_vals, axis=0)

    scaler_params = load_scaler_params(scaler_save_path)

    test_predictions_original = manual_inverse_transform(
        test_predictions.squeeze().reshape(1, -1), scaler_params)

    test_true_vals_original = manual_inverse_transform(
        test_true_vals.squeeze().reshape(1, -1), scaler_params)

    print(
        f"Length of test_predictions_original: {len(test_predictions_original)}"
    )
    print(f"Length of test_true_vals_original: {len(test_true_vals_original)}")
    print(f"Length of test_dates: {len(test_future_dates)}")

    test_df = pd.DataFrame({
        "index":
        index,
        "date":
        test_future_dates,
        "actual":
        list(np.squeeze(test_true_vals_original).flatten())
    })
    test_df.set_index("date", inplace=True)

    predictions = pd.DataFrame({
        "date":
        test_future_dates,
        "prediction":
        np.squeeze(test_predictions_original.reshape(-1))
    })
    predictions.set_index("date", inplace=True)

    test_monthly_actual = test_df["actual"]
    test_monthly_forecast = predictions["prediction"]

    metrics = evaluation(test_monthly_actual.values,
                         test_monthly_forecast.values, args.future_steps)

    resampled_df = pd.DataFrame({
        "date": test_monthly_actual.index,
        "actual": test_monthly_actual.values,
        "forecast": test_monthly_forecast.values,
    })
    print("test_monthly_results", resampled_df)
    resampled_df["date"] = pd.to_datetime(resampled_df["date"])

    resampled_df = resampled_df[(resampled_df["date"] >= args.test_eval_start)
                                & (resampled_df["date"] <= args.test_eval_end)]

    resampled_df.to_csv(args.save_path + "/monthly_results.csv", index=False)

    resampled_metrics = evaluation(resampled_df["actual"].values,
                                   resampled_df["forecast"].values,
                                   args.future_steps)
    print("resampled_metrics", resampled_metrics)

    with open(args.save_path + "/resampled_metrics.json", "w") as f:
        json.dump(serialize_dict(resampled_metrics), f)

    with open(args.save_path + "/train_losses.json", "w") as f:
        json.dump(train_losses, f)

    with open(args.save_path + "/val_losses.json", "w") as f:
        json.dump(val_losses, f)

    print("Evaluation metrics for resampled data saved successfully.")
    return resampled_metrics


def get_current_script_name():

    return os.path.basename(__file__)


def get_aug_bool():
    scriptn = get_current_script_name()
    if "_Aug.py" in scriptn:
        return True
    else:
        return False


def main(
    config,
    args,
    df,
    df_news,
    model_name,
):
    result = defaultdict(dict)
    result[model_name] = train_model(config, args, df, model_name)
    return result, config


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    AUGMENT = get_aug_bool()
    parser = argparse.ArgumentParser(description=" training script")
    parser.add_argument("--MODEL", type=str, default="CNNLSTM")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=512,
                        help="Number of hidden_dim for the model")
    parser.add_argument("--past_steps",
                        type=int,
                        default=10,
                        help="Input window ")
    parser.add_argument("--future_steps", type=int, default=1, help=" window ")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=1000,
                        help="Number of training epochs")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.6,
                        help="Dropout rate")
    parser.add_argument("--n_features", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--SEED", type=int, default=100)
    parser.add_argument("--patience", default=1)
    parser.add_argument("--dtype",
                        default="float",
                        choices=["float", "double"])
    parser.add_argument("--time_col", default="date")
    parser.add_argument("--target_col", default="total_index")
    parser.add_argument("--augment", type=str2bool, default=AUGMENT)
    parser.add_argument("--sentiment", type=str2bool, default=True)
    parser.add_argument("--test_start", default="2019-05-01")
    parser.add_argument("--test_end", default="2021-10-31")
    parser.add_argument("--test_eval_start", default="2020-04-01")
    parser.add_argument("--test_eval_end", default="2021-09-01")
    args = parser.parse_args()
    set_seed(args.SEED)
    args.total_features = args.n_features
    args.MODEL = f"{args.MODEL}-F{str(args.n_features).zfill(2)}"
    if args.sentiment:
        args.MODEL += "-Senti"
        args.total_features += 1
    else:
        args.MODEL += "-XSenti"

    if args.augment:
        args.MODEL += "-AUG"
        args.past_steps = 310
        args.future_steps = 31
    else:
        args.MODEL += "-XAUG"

    args.MODEL += f"-d{args.hidden_dim}"
    print("args.MODEL", args.MODEL)
    print("args.total_features", args.total_features)
    args.sentiment
    print("args.sentiment", args.sentiment)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
    print("Available devices ", torch.cuda.device_count())

    f = "../data/data_config.json"
    with open(f, "r") as ff:
        data_config = json.load(ff)

    args.device = torch.device("cuda:0")

    args.SEED = 100
    args.save_path = os.path.join("../results/", "cpi", args.MODEL,
                                    str(args.SEED))
    os.makedirs(args.save_path, exist_ok=True)
    config = vars(args)
    config, args = retrieve_last_processed_give_lc_path(config, args)
    config.update(data_config)
    if args.augment:
        fpath_cpi = config["df_fpath_aug"]
    else:
        fpath_cpi = config["df_fpath_xaug"]
    df = pd.read_csv(fpath_cpi, encoding="utf-8-sig")
    df_news = df

    results, config = main(config,
                            args,
                            df,
                            df_news,
                            model_name=args.MODEL)

    results_df = pd.DataFrame([results[args.MODEL]])
    results_df.to_csv(os.path.join(args.save_path, "multi_result.csv"),
                        encoding="utf-8-sig",
                        index=False)

    with open(os.path.join(args.save_path, "multi_result.json"),
                "w") as json_file:
        json.dump(serialize_dict(results[args.MODEL]), json_file)

    print("Results saved as multi_result.csv and multi_result.json")

    save_path = config["save_path"]
    save_to_json(config, os.path.join(save_path, "config.json"))
