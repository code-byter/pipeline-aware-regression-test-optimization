import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer


def preprocess_es_dataset(df):
    # Filter Test results
    df = df[~df.test_result.str.contains("INFO")]

    # Test name formatting
    df.test_name = df.test_name.str.replace("_", "/")
    df["flaky"] = False

    dfs = dict(tuple(df.groupby("test_name")))

    for key in list(dfs.keys()):
        df_tmp = dfs[key]
        for idx in range(2, len(df_tmp) - 5):
            if df_tmp.test_result.iloc[idx] == "FLAKY":
                df.loc[df_tmp.index[idx], "flaky"] = True
                df.loc[df_tmp.index[idx + 1], "flaky"] = True

            # No change
            if df_tmp.test_result.iloc[idx] == df_tmp.test_result.iloc[idx - 1]:
                continue

            # States are the same
            if (
                df_tmp.test_result.iloc[idx]
                == df_tmp.test_result.iloc[idx + 1]
                == df_tmp.test_result.iloc[idx + 2]
                == df_tmp.test_result.iloc[idx + 3]
            ):
                continue

            df.loc[df_tmp.index[idx], "flaky"] = True
            df.loc[df_tmp.index[idx + 1], "flaky"] = True
            df.loc[df_tmp.index[idx + 2], "flaky"] = True
            df.loc[df_tmp.index[idx + 3], "flaky"] = True

    return df


def scale_dataset(df, scaling_params):
    [vectorizer, pca, scaler] = scaling_params

    # Bag of word model for test name
    X_word = vectorizer.transform(list(df.test_name))

    # PCA
    nr_components = 9
    df[[f"PCA_{idx}" for idx in range(nr_components)]] = pca.transform(X_word.toarray())

    # StandardScaler
    scaled_features = ["test_duration"]
    df[scaled_features] = scaler.transform(df[scaled_features])

    return df


def generate_scaler_params(df):
    # Bag of word model for test name
    vectorizer = CountVectorizer(stop_words=["bmw", "test"], max_features=55)
    X_word = vectorizer.fit_transform(list(df.test_name))

    # PCA
    nr_components = 9
    pca = PCA(n_components=nr_components)
    pca.fit(X_word.toarray())

    # StandardScaler
    scaled_features = ["test_duration"]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df[scaled_features])

    return [vectorizer, pca, scaler]


if __name__ == "__main__":
    df = pd.read_csv("pre_submit_dataset.csv")
    df = preprocess_es_dataset(df)
    [vectorizer, pca, scaler] = generate_scaler_params(df)
    df = scale_dataset(df, [vectorizer, pca, scaler])
    df.to_pickle("dataset/preprocessed_dataset_pre_submit_public.pkl")

    df = pd.read_csv("post_submit_dataset.csv")
    df = preprocess_es_dataset(df)
    [vectorizer, pca, scaler] = generate_scaler_params(df)
    df = scale_dataset(df, [vectorizer, pca, scaler])
    df.to_pickle("dataset/preprocessed_dataset_post_public.pkl")
