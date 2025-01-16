import pandas as pd


def multi_aug(config, args, data, method="linear"):
    d = data.copy()
    col_name = list(set(d.columns) - set([args.time_col]))
    d[col_name] = d[col_name].astype("float32")
    d[args.time_col] = pd.to_datetime(d[args.time_col])
    d = d.set_index(args.time_col)
    d = d.resample("D").interpolate(method=method).dropna()
    return d


def multi_aug_linear(data):
    d = data.copy()
    col_name = list(set(d.columns) - set([args.time_col]))
    d[col_name] = d[col_name].astype("float32")
    d[args.time_col] = pd.to_datetime(d[args.time_col])
    d = d.set_index(args.time_col)
    d = d.resample("D").interpolate(method="spline", order=2).dropna()
    return d


def restore_monthly_points(test_set_scaled, start_date):

    test_dates = pd.date_range(start=start_date,
                               periods=len(test_set_scaled),
                               freq="D")

    test_df = pd.DataFrame(
        test_set_scaled,
        index=test_dates,
        columns=[
            "total_index",
        ],
    )

    test_restored_monthly = test_df.resample("MS").first()

    return test_restored_monthly


def uni_aug(data):
    d = data.copy()
    col_name = list(set(d.columns) - set(["period"]))
    d[col_name] = d[col_name].values.astype("float32")
    d["period"] = pd.to_datetime(d["period"])

    d = d.squeeze()

    d = d.set_index("period")

    upsampled = d.resample("D")

    interpolated = upsampled.interpolate(method="spline", order=2)
    interpolated = interpolated.dropna()
    return interpolated
