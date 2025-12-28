def create_target(df):
    df = df.copy()
    df["target"] = df["ret"].shift(-1)
    return df.dropna().reset_index(drop=True)
