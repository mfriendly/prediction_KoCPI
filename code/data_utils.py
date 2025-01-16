import argparse
import csv
import glob
import json
import os
import pickle
import string
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas import DatetimeIndex
from data_utils import *
from training_utils import *

import logging
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")

def load_scaler_params(scaler_path):
    with open(scaler_path, "r") as f:
        scaler_params = json.load(f)
    return scaler_params


def slicing(df):
    total_length = len(df)
    train_idx = int(0.7 * total_length)
    val_idx = int(0.8 * total_length)

    train_set = df[:train_idx]
    val_set = df[train_idx:val_idx]
    test_set = df[val_idx:]

    return train_set, val_set, test_set


def slicing_by_date(df):

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    train_start = "2010-01-01"
    train_end = "2018-04-30"
    val_start = "2018-05-01"
    val_end = "2019-04-30"
    test_start = "2019-05-01"
    print("test_start", test_start)
    test_end = "2021-10-31"
    print("test_end", test_end)
    trainval_set = df.loc[:val_end]
    train_set = df.loc[train_start:train_end]
    val_set = df.loc[val_start:val_end]
    test_set = df.loc[test_start:test_end]

    return train_set, val_set, test_set, trainval_set


def pad_and_average(lists, max_length=500):

    padded_lists = [
        np.pad(lst, (0, max_length - len(lst)),
               "constant",
               constant_values=np.nan) for lst in lists
    ]

    padded_array = np.array(padded_lists)

    average_values = np.nanmean(padded_array, axis=0)

    return average_values


def manual_inverse_transform(scaled_data, scaler_params):

    min_vals = np.array(scaler_params["min"])[0]
    max_vals = np.array(scaler_params["max"])[0]

    original_data = scaled_data * (max_vals - min_vals) + min_vals

    print("original_data", original_data)
    return original_data


def get_start_end_dates(data):
    """
    This function returns the start and end date for a given dataset.
    """
    start_date = data.index.min()
    end_date = data.index.max()
    return start_date, end_date


def calculate_statistics(data, columns):
    statistics = {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "median": np.median(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
        "count": np.count_nonzero(~np.isnan(data), axis=0),
    }

    stats_df = pd.DataFrame(statistics, index=columns[1:])
    return stats_df


def save_statistics_to_csv(statistics_df, filename):
    statistics_df.to_csv(filename, index=True)


def save_scaler_params(scaler, save_path):
    scaler_params = {
        "min": scaler.data_min_.tolist(),
        "max": scaler.data_max_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min_max_range": scaler.min_.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(scaler_params, f)


def create_embedding_sizes(config, embedding_dim, df,
                           time_varying_categorical_variables):
    embedding_sizes = {}
    for col in time_varying_categorical_variables:
        df[col] = df[col].astype("int64").fillna(99)
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].astype("int").astype("str").astype("category")
        num = df[col].nunique() + 1
        embedding_sizes[col] = (int(num), int(embedding_dim))

    config["x_categorical_embed_sizes"] = embedding_sizes
    return config


def plot_shared_x_two_y(args):
    values_1_path = f"../code/z_csv/{args.nation}_var_importance_final_values.npy"
    labels_shared_path = f"../code/z_csv/{args.nation}_var_importance_final_labels.json"
    csv_path2 = "../code/z_visualizations/plt_Hparam/heatmap_6.csv"

    values_1 = np.load(values_1_path)
    with open(labels_shared_path, "r") as f:
        labels = json.load(f)
    labels.reverse()
    labels_w_label = ["new_confirmed"] + labels

    values_1 = [0] + list(values_1)[::-1]

    df2 = pd.read_csv(csv_path2)
    values_df2 = list(df2["8"].values)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars = ax1.bar(labels_w_label,
                   values_1,
                   color="skyblue",
                   edgecolor="black",
                   linewidth=0.5)
    ax1.set_xlabel("Labels")
    ax1.set_ylabel("Importance", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax1.set_ylim(bottom=10)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 yval,
                 round(yval, 2),
                 ha="center",
                 va="bottom")

    ax2 = ax1.twinx()
    ax2.plot(df2["Top_Variables"], values_df2, color="red")
    ax2.set_ylabel("Heatmap Value", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(bottom=min(values_df2) * 0.999)

    ax1.set_xticklabels(labels_w_label, rotation=90)

    plt.title("Bar and Line Plot with Shared X-Axis")
    fig.tight_layout()

    plt.savefig("shared_with_outline2.png")
    plt.close()


def random_combination_of_vars(all_vars):

    random_length = random.randint(1, len(all_vars))

    random_combo = random.sample(all_vars, random_length)
    print("random_combo", random_combo)

    return random_combo


def get_nation():

    current_dir = os.getcwd()
    grandparent_dir = os.path.basename(os.path.dirname(current_dir))

    if "US" in grandparent_dir:
        return "US"
    elif "UK" in grandparent_dir:
        return "UK"
    else:
        return "Unknown"


def sparse_matrix_save_npy(args, matrix, index, name):
    matrix = matrix.squeeze().cpu().numpy()
    dir_ = f"{args.save_path}/_{name}"
    os.makedirs(dir_, exist_ok=True)
    np.save(f"{args.save_path}/_{name}/_matrix__{str(index).zfill(2)}.npy",
            matrix)


def adap_matrix_save_npy_png(args, matrix, index, time_step, name):

    matrix = matrix.squeeze().cpu().numpy()
    dir_ = f"{args.save_path}/_{name}"
    os.makedirs(dir_, exist_ok=True)
    np.save(
        f"{args.save_path}/_{name}/_matrix__{str(index).zfill(2)}_{str(time_step).zfill(2)}.npy",
        matrix,
    )
    if time_step == 42:
        plt.figure(figsize=(8, 8))
        sns.heatmap(matrix,
                    cmap="GnBu",
                    square=True,
                    linecolor="white",
                    linewidths=0.5)

        plt.savefig(
            f"{args.save_path}/_{name}/_matrix__{str(index).zfill(2)}_{str(time_step).zfill(2)}.png",
            dpi=300,
        )

        plt.close()


def update_args_odd_input_size(args):
    if (len(args.input_cols) - 2) % 2 == 0:
        args.odd = False

    else:
        args.odd = True
    return args


def setup_logger(name=__name__, log_file=None, level=logging.INFO):
    """Function to set up a logger with a specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def build_loader(config, dataset_, SELECT_BATCHSIZE, device):
    return DataLoader(
        dataset_,
        batch_size=SELECT_BATCHSIZE,
        num_workers=0,
        shuffle=True,
        generator=torch.Generator(device="cpu"),
        drop_last=False,
    )


def create_loaders_from_dataset(config, train_dataset, val_dataset,
                                SELECT_BATCHSIZE, device):
    train_loader = build_loader(config, train_dataset, SELECT_BATCHSIZE,
                                device)

    val_loader = build_loader(config, val_dataset, SELECT_BATCHSIZE * 3,
                              device)
    return train_loader, val_loader


def set_save_paths(config, args):
    base_save_path = config["save_path"]
    os.makedirs(base_save_path, exist_ok=True)
    if args.pretrain in ["contrastive", "supervised"]:
        pretrain_path = os.path.join(base_save_path, "pretrain")
        config["save_path"] = pretrain_path
        os.makedirs(pretrain_path, exist_ok=True)

    if args.finetune:
        finetune_path = os.path.join(base_save_path, "finetune")
        config["save_path"] = finetune_path
        os.makedirs(finetune_path, exist_ok=True)

    return config


def update_inputs_config(config):
    input_cols = config["input_cols"]
    holiday_cols = config["holiday_cols"]

    time_cat_cols = config["time_cat_cols"]
    policy_lag_vars = config["policy_lag_vars"]
    time_varying_categorical_variables = holiday_cols + time_cat_cols + policy_lag_vars

    config["static_variables_len"] = 0

    config["input_size"] = len(input_cols) - 2

    time_varying_real_variables_encoder = config["unknown_real_x_target"]

    config["x_categoricals"] = time_varying_categorical_variables
    config["time_varying_categorical_variables_len"] = len(
        time_varying_categorical_variables)
    config["time_varying_real_variables_encoder_len"] = len(
        time_varying_real_variables_encoder)
    config[
        "time_varying_categorical_variables_list"] = time_varying_categorical_variables
    config[
        "time_varying_real_variables_encoder_list"] = time_varying_real_variables_encoder
    config["input_cols"] = input_cols
    config["time_varying_real_variables_decoder_len"] = len(
        time_varying_real_variables_encoder) - 1

    config["rnn_hidden_dimension"] = config["embedding_dim"] * (
        len(time_varying_real_variables_encoder) +
        len(time_varying_categorical_variables))
    config["num_masked_series"] = 1
    return config


def window_evaluation_and_append_metrics(
    config,
    actuals,
    inverse_transformed_predictions,
    predictions_to_scale,
    outputs,
    node_id,
    abbr,
    week_metrics_list,
):

    RMSE_window_metric, MAE_window_metric, MAPE_window_metric, CORR_window_metric = window_evaluation(
        config,
        actuals,
        inverse_transformed_predictions,
        predictions_to_scale.squeeze().reshape(1, -1),
        outputs[:,
                node_id, :, :].cpu().detach().numpy().squeeze().reshape(1, -1),
        node_id,
    )
    result_list = []
    for i, week_metrics in enumerate(week_metrics_list):

        result = append_window_metrics(
            week_metrics,
            abbr,
            RMSE_window_metric[i],
            MAE_window_metric[i],
            MAPE_window_metric[i],
            CORR_window_metric[i],
        )

        result_list.append(result)

    return result_list


def save_metrics_and_log_results(
    config,
    args,
    node_id,
    index,
    week_metrics_list,
    abbr,
    save_path,
    MODEL,
    csv_file_name,
    mean_wk,
):
    if node_id > args.num_nodes - 2 and index == 3:
        metric_dicts_and_filenames = [
            (week_metrics_list[0], "window_metric_1wk.json"),
            (week_metrics_list[1], "window_metric_2wk.json"),
            (week_metrics_list[2], "window_metric_3wk.json"),
            (week_metrics_list[3], "window_metric_4wk.json"),
            (week_metrics_list[4], "window_metric_5wk.json"),
            (week_metrics_list[5], "window_metric_6wk.json"),
        ]

        for metric_dict__, filename in metric_dicts_and_filenames:
            for abbr__ in metric_dict__:
                RMSE_mean = np.mean(metric_dict__[abbr__]["RMSE_LIST"])
                CORR_mean = np.mean(metric_dict__[abbr__]["CORR_LIST"])
                MAPE_mean = np.mean(metric_dict__[abbr__]["MAPE_LIST"])
                MAE_mean = np.mean(metric_dict__[abbr__]["MAE_LIST"])

                metric_dict__[abbr__]["RMSE"] = RMSE_mean
                metric_dict__[abbr__]["CORR"] = CORR_mean
                metric_dict__[abbr__]["MAPE"] = MAPE_mean
                metric_dict__[abbr__]["MAE"] = MAE_mean

                file_path = os.path.join(save_path, filename)

                with open(file_path, "w", encoding="utf-8") as json_file:
                    json.dump(serialize_dict(metric_dict__),
                              json_file,
                              indent=4)

                if mean_wk in filename and not np.isnan(RMSE_mean):
                    print("mean_wk", mean_wk)
                    csv_output_file = csv_file_name
                    with open(csv_output_file,
                              mode="a",
                              newline="",
                              encoding="utf-8") as file:
                        writer = csv.writer(file)
                        name = MODEL
                        writer.writerow([
                            abbr__, name, RMSE_mean, MAE_mean, MAPE_mean,
                            CORR_mean
                        ])
                        print(
                            "State, Model, RMSE, MAE, MAPE, CORR",
                            abbr__,
                            name,
                            RMSE_mean,
                            MAE_mean,
                            MAPE_mean,
                            CORR_mean,
                        )
                        print("file_path", file_path)


def process_and_save_metrics(
    config,
    args,
    index,
    actuals,
    inverse_transformed_predictions,
    predictions_to_scale,
    outputs,
    node_id,
    abbr,
    week_metrics_list,
    save_path,
    MODEL,
    file,
    mean_wk,
):
    config["index"] = index
    week_metrics_list = window_evaluation_and_append_metrics(
        config,
        actuals,
        inverse_transformed_predictions,
        predictions_to_scale,
        outputs,
        node_id,
        abbr,
        week_metrics_list,
    )

    save_metrics_and_log_results(
        config,
        args,
        node_id,
        index,
        week_metrics_list,
        abbr,
        save_path,
        MODEL,
        file,
        mean_wk,
    )


def append_window_metrics(week_metrics, abbr, RMSE, MAE, MAPE, CORR):

    week_metrics[abbr]["MAPE_LIST"].append(MAPE)
    week_metrics[abbr]["RMSE_LIST"].append(RMSE)
    week_metrics[abbr]["CORR_LIST"].append(CORR)
    week_metrics[abbr]["MAE_LIST"].append(MAE)
    return week_metrics


def inverse_transform_predictions(config, args, predictions_to_scale, abbr,
                                  standard_scaler_stats_dict):
    if args.scaler_name == "":
        mean = standard_scaler_stats_dict[abbr]["mean"]
        std = standard_scaler_stats_dict[abbr]["std"]
        inverse_transformed_predictions = (predictions_to_scale * std) + mean
        return inverse_transformed_predictions
    return predictions_to_scale


def replace_negatives_and_interpolate(arr):

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    arr = np.where(arr <= 0.0, np.nan, arr)
    nans, x = nan_helper(arr)
    arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])

    return arr


def update_predictions(
    preds_dict,
    abbr,
    start_date,
    end_date,
    inverse_transformed_predictions,
    combined_tar,
    past_steps,
    dates,
    save_path,
):
    preds_dict.setdefault(abbr, {}).setdefault(start_date, {})

    preds_dict[abbr][start_date] = {
        "PRED": inverse_transformed_predictions.squeeze().reshape(1, -1),
        "ACTUAL_past": combined_tar[:past_steps, :].reshape(1, -1),
        "ACTUAL_future": combined_tar[past_steps:, :].reshape(1, -1),
        "DATES_past": dates[:past_steps],
        "DATES_future": dates[past_steps:],
        "start_date": start_date,
        "end_date": end_date,
    }

    save_to_json(preds_dict, f"{save_path}/preds.json")

    return preds_dict


def apply_mean_calculation(config, args, csv_file_name):
    column_names = [
        "abbr", "name", "RMSE_mean", "MAE_mean", "MAPE_mean", "CORR_mean"
    ]

    eval_df = pd.read_csv(csv_file_name,
                          header=None,
                          names=column_names,
                          encoding="utf-8")

    averages = eval_df.iloc[:, 2:6].mean(axis=0)
    assert eval_df.shape[0] == args.num_nodes
    averages_df = pd.DataFrame(
        [averages],
        columns=["RMSE_mean", "MAE_mean", "MAPE_mean", "CORR_mean"])

    print("averages_df", averages_df)

    output_file_name = csv_file_name.replace(".csv", "_mean.csv")

    averages_df.to_csv(output_file_name, index=False, encoding="utf-8")

    return output_file_name


def save_normalized_similarity_matrix(embeddings, output_file):
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1),
                                            embeddings.unsqueeze(0),
                                            dim=2)
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()
    normalized_similarity_matrix = (similarity_matrix - min_val) / (max_val -
                                                                    min_val)
    normalized_similarity_matrix_np = normalized_similarity_matrix.numpy()
    np.save(output_file, normalized_similarity_matrix_np)


def read_and_resave_pkl_cpu(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, torch.Tensor):
        data = data.to("cpu")

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_matrix(file_path, name, dtype=torch.float64):
    try:
        matrix = np.load(file_path)
        return torch.tensor(matrix, dtype=dtype)
    except Exception as e:
        print(f"Failed to process {name} matrix: {e}")
        return None


def load_matrices_and_process(input_path, nation, config, device):

    graph_files = {
        "dist": f"{input_path}/x_data_aux/{nation}/matrix_0.npy",
        "travel": f"{input_path}/x_data_aux/{nation}/matrix_1.npy",
    }

    gdict = {
        name: load_matrix(file_path, name)
        for name, file_path in graph_files.items()
    }

    adjs = []
    for gg in config["GRAPHS"]:
        matrix = gdict.get(gg)
        if matrix is not None:
            try:
                adjs.append(matrix.to(device))
            except Exception as e:
                print(f"Failed to move graph '{gg}' to device: {e}")
        else:
            print(f"Graph type '{gg}' is not available.")

    return adjs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def retrieve_pandas_date_list(start_date_, end_date_):
    return pd.date_range(start=start_date_, end=end_date_, freq="D")


def retrieve_start_end_dates_of_batch(batch):
    start_date = batch["start_date"].detach().cpu().numpy().item()
    start_date_ = format_date(start_date)
    end_date = batch["end_date"].detach().cpu().numpy().item()
    end_date_ = format_date(end_date)
    return start_date_, end_date_


def retrieve_metadata_of_nation(config, nation, input_path, device):
    df_mapper = pd.read_csv(input_path +
                            f"/x_data_aux/{nation}/statemappings{nation}.csv")
    list_of_states = df_mapper["State"].tolist()
    list_of_abbr = df_mapper["Abbr"].tolist()
    states2abbr = dict(zip(list_of_states, list_of_abbr))
    try:
        adjs = load_matrices_and_process(input_path, nation, config, device)
    except:
        adjs = None
    return list_of_states, list_of_abbr, states2abbr, adjs


def create_result_dictionaries(nation, list_of_abbr):

    preds_dict = {ab: {} for ab in list_of_abbr}
    window_metric_1wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_2wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_3wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_4wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_5wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    window_metric_6wk = {
        ab: {
            "RMSE": None,
            "MAPE": None,
            "CORR": None,
            "MAE": None,
            "RMSE_LIST": [],
            "CORR_LIST": [],
            "MAPE_LIST": [],
            "MAE_LIST": [],
        }
        for ab in list_of_abbr
    }
    week_metrics_list = [
        window_metric_1wk,
        window_metric_2wk,
        window_metric_3wk,
        window_metric_4wk,
        window_metric_5wk,
        window_metric_6wk,
    ]
    return week_metrics_list, preds_dict


def retrieve_nation_args(args, nation):
    if nation == "US":
        args.nation = nation

        args.GRAPHS = ["dist", "travel"]
    if nation == "US":
        args.nation = nation

        args.GRAPHS = ["dist", "travel"]
    elif nation == "US":
        args.nation = nation

        args.GRAPHS = ["dist", "travel"]
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def count_param_and_update_config(config, model_):

    model_parameters = filter(lambda p: p.requires_grad, model_.parameters())

    params = sum(p.numel() for p in model_parameters)

    config["PARAMS"] = params
    return config


def load_matrices(input_path, nation):
    travel_matrix = np.load(input_path + f"/x_data_aux/{nation}/matrix_1.npy")
    dist_matrix = np.load(input_path + f"/x_data_aux/{nation}/matrix_0.npy")
    dist_matrix__ = torch.tensor(dist_matrix, dtype=torch.float64)
    travel_matrix__ = torch.tensor(travel_matrix, dtype=torch.float64)
    return dist_matrix__, travel_matrix__


def prepare_adjs(GRAPHS, gdict, device):
    adjs = []
    for gg in GRAPHS:
        adjs.append(gdict[gg].to(device))
    return adjs


def save_last_processed_number(save_path, number):
    with open(save_path + "/last_processed.txt", "w") as file:
        file.write(str(number))


def get_last_processed_number(path):
    files = glob.iglob(os.path.join(path, "*.pth"))
    max_int = -1
    nums = []
    for file_path in files:
        base_name = os.path.basename(file_path)
        number = base_name.split("-v")[1].replace(".pth", "")
        print("number", number)
        nums.append(int(number))
    if nums:
        max_int = max(max_int, max(nums))
    return max_int if max_int != -1 else 0


def retrieve_last_processed_give_lc_path(config, args):
    CKPT_SAVE = config["save_path"] + "/y_ckpt/"
    if not os.path.exists(CKPT_SAVE):
        os.makedirs(CKPT_SAVE)
    os.chmod(CKPT_SAVE, 0o700)
    last_processed_num = int(get_last_processed_number(CKPT_SAVE))

    last_processed_num = str(last_processed_num)
    C_PATH_num = int(last_processed_num) + 1
    save_last_processed_number(CKPT_SAVE, last_processed_num)
    L_PATH = CKPT_SAVE + f"bests-v{last_processed_num}.pth"
    C_PATH = CKPT_SAVE + f"bests-v{C_PATH_num}.pth"
    config["CKPT_SAVE"] = CKPT_SAVE
    config["L_PATH"] = L_PATH
    config["C_PATH"] = C_PATH
    return config, args


def save_pkl(device, address, **dataframes):

    for suffix, df in dataframes.items():
        filename = f"{address}{suffix}.pkl"
        df.to_pickle(filename)


def load_pkl(device, address, *suffixes):
    dataframes = {}
    for suffix in suffixes:
        filename = f"{address}{suffix}.pkl"
        df = pd.read_pickle(filename)
        dataframes[suffix] = df
    return dataframes


def thousand_separator(x, pos):
    return f"{x:,.0f}"


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def reconstruct_series_to_array(
    differences,
    initial_values,
    start_date_,
    end_date_,
    date_format="%Y-%m-%d",
    abbr=None,
    days=1,
    future_steps=42,
):
    start_date = datetime.strptime(start_date_, date_format) if isinstance(
        start_date_, str) else start_date_
    start_date = start_date + timedelta(days=future_steps)
    end_date = datetime.strptime(end_date_, date_format) if isinstance(
        end_date_, str) else end_date_
    dates = pd.date_range(start_date, end_date)
    reconstructed_values = [
        initial_values[(start_date -
                        timedelta(days=days)).strftime(date_format)]
        for start_date in dates
    ]
    for i in range(len(differences)):
        new_value = reconstructed_values[i] + differences[i][0]
        reconstructed_values[i] = new_value
    reconstructed_values = np.array(reconstructed_values).reshape(-1, 1)
    return reconstructed_values


def difference_from_previous_time(df,
                                  column_name="new_confirmed_Y",
                                  list_of_abbr=None,
                                  abbr=None,
                                  days=1):
    date_column_name = "date"
    initial_values = df[[date_column_name, column_name
                         ]].set_index(date_column_name)[column_name].to_dict()
    start_date = datetime.strptime("2021-09-12", "%Y-%m-%d")
    end_date = datetime.strptime("2022-03-01", "%Y-%m-%d")
    initial_values = {
        key.strftime("%Y-%m-%d"): value
        for key, value in initial_values.items()
        if start_date <= key <= end_date
    }
    df[f"{column_name}_time_diff"] = df[column_name] - df[column_name].shift(
        days)
    df[f"{column_name}_time_diff"] = df[f"{column_name}_time_diff"].fillna(0.0)
    df[column_name] = df[f"{column_name}_time_diff"]
    return df, initial_values


def interpolate_data(df, state, epidemic_column="new_confirmed"):
    state_df = df[df["State"] == state].copy()
    for i, row in state_df.iterrows():
        if (i % 7) != 0:
            state_df.at[i, epidemic_column] = np.nan
    state_df[epidemic_column] = state_df[epidemic_column].interpolate(
        method="linear")
    df.loc[state_df.index, epidemic_column] = state_df[epidemic_column]
    return df


def replace_cols(df, dict_):
    df.rename(columns=dict_)
    return df


def get_values_in_date_range(csv_path,
                             start_date,
                             end_date,
                             value_column="new_confirmed"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    filtered_df = df.loc[mask]
    values = filtered_df["new_confirmed"]
    values_array = values.to_numpy()
    return values_array


def smoothing(df, real_columns):
    df1 = df.copy()
    df0 = df1.iloc[:6]
    df0.loc[:6, real_columns] = np.nan
    df1 = df1.iloc[6:]
    for col in real_columns:
        temp = []
        for i in range(6, len(df)):
            ave = np.mean(df[col].iloc[i - 6:i + 1])
            temp.append(ave)
        df1[col] = temp
        try:
            df1.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            pass
        try:
            df1.drop(["level_0"], axis=1, inplace=True)
        except:
            pass
    df2 = pd.concat([df0, df1], axis=0)
    df2 = df2.bfill()
    return df2


def save_pickle(file, filename):
    with open(filename, "wb") as f:
        pickle.dump(file, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        file = pickle.load(f)
    return file


def timestamp_to_unix(timestamp):
    try:
        return timestamp.value // 10**9
    except:
        return pd.to_datetime(timestamp).value // 10**9


def format_date(unix_timestamp):
    if isinstance(unix_timestamp, int):
        try:
            dt = datetime.fromtimestamp(unix_timestamp)
        except Exception:

            return None
    elif isinstance(unix_timestamp, datetime.datetime):
        dt = unix_timestamp
    else:
        return None
    return f"{dt.year}-{dt.month:02d}-{dt.day:02d}"


def serialize_dict(data):
    if isinstance(data, dict):
        return {k: serialize_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_dict(item) for item in data]
    elif isinstance(data, np.ndarray):
        return serialize_dict(data.tolist())
    elif isinstance(data, torch.device):
        return str(data)

    elif isinstance(
            data,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(data)
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, DatetimeIndex):
        return data.strftime("%Y-%m-%d").tolist()
    return data


def save_to_json(dict_, name):
    with open(name, "w") as f:
        json.dump(serialize_dict(dict_), f, indent=4)


def generate_next_identifier(current_index):
    base = len(string.ascii_uppercase)
    result = ""
    while True:
        quotient, remainder = divmod(current_index, base)
        result = string.ascii_uppercase[remainder] + result
        if quotient == 0:
            break
        current_index = quotient - 1
    return result


def interpolate_np(arr):
    indices = np.arange(len(arr))
    not_nan = ~np.isnan(arr)[0]
    linear_interpolator = interpolate.interp1d(indices[not_nan],
                                               arr[not_nan],
                                               kind="linear",
                                               fill_value="extrapolate")
    arr_interpolated = linear_interpolator(indices)
    return arr_interpolated


def numpy_ffill(arr):
    arr = arr.squeeze()
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out


def load_json(json_file_name):
    with open(json_file_name, "r") as json_file:
        iii = json.load(json_file)
    return iii


def check_if_exp_id_exists_from_reading_json_then_create_one_if_not(
        input_cols, input_path, nation):
    json_file_name = input_path + f"/x_data_aux/{nation}/experiment_ids.json"
    try:
        with open(json_file_name, "r") as json_file:
            experiment_ids = json.load(json_file)
    except FileNotFoundError:
        experiment_ids = {}
    subset_key = "_".join(sorted(input_cols))
    if subset_key in experiment_ids:
        return experiment_ids[subset_key]
    else:
        next_identifier = generate_next_identifier(len(experiment_ids))
        new_experiment_id = f"{len(input_cols)}{next_identifier}"
        experiment_ids[subset_key] = new_experiment_id
        with open(json_file_name, "w") as json_file:
            json.dump(experiment_ids, json_file, indent=4)
        return new_experiment_id


def join_paths(p1, p2):
    p3 = os.path.join(p1, p2)
    return p3


def change_config_for(config, args):

    config["stage"] = "finetune"
    save_path = config["save_path"] = config["save_path"].replace(
        "/pretrain", "/finetune")

    config["C_PATH"] = config["C_PATH"].replace("pretrain", "finetune")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return config, args


def get_current_script_name():

    return os.path.basename(__file__)


def get_aug_bool():
    scriptn = get_current_script_name()
    if "_Aug.py" in scriptn:
        return True
    else:
        return False


if __name__ == "__main__":

    plot_shared_x_two_y()
