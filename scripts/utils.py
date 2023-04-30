import numpy as np
import pandas as pd

def convert_to_arr(input_str, dim):
    if dim == 1:
        input_str = input_str[1:-1]
    elif dim == 2:
        input_str = input_str.replace(" ", "")  # Remove spaces
        input_str = input_str.replace("][", "],[")  # Add commas between sub-arrays
    else:
        raise NotImplementedError

    # Evaluating the string to create a nested list
    nested_list = eval(input_str)

    # Converting the nested list to a numpy array
    np_array = np.array(nested_list)
    return np_array


def get_id(url):
    url_parts = url.split("?")

    # Extracting the query parameters part of the URL
    query_params = url_parts[1] if len(url_parts) > 1 else ""

    # Splitting the query parameters by "&" to separate individual key-value pairs
    query_params = query_params.split("&")

    # Extracting the video ID from the query parameters
    video_id = ""
    for param in query_params:
        if param.startswith("v="):
            video_id = param[2:]
        break
    return video_id


data_pkl = {}
df_vid = pd.read_csv("../data/master_features.csv", header=0, names=["id", "emb","keystep"])
df_text = pd.read_csv("../youcook2/reviewed_0812.csv")

for i in range(df_vid.shape[0]):
    vid_name, vid_arr, vid_key = df_vid.iloc[i][0], convert_to_arr(df_vid.iloc[i][1],dim=2), convert_to_arr(df_vid.iloc[i][2],dim=1)
    matches = df_text["VideoID"].str.contains(vid_name)
    data_pkl[vid_name] = {}
    data_pkl[vid_name]["Sentence"] = df_text.loc[matches]["Sentence"].values
    data_pkl[vid_name]["IsUsefulSentence"] = df_text.loc[matches]["IsUsefulSentence"].values
    data_pkl[vid_name]["Key steps"] = df_text.loc[matches]["Key steps"].values
    data_pkl[vid_name]["Verb"] = df_text.loc[matches]["Verb"].values
    data_pkl[vid_name]["Object"] = df_text.loc[matches]["Object(directly related with Verb)"].values
    data_pkl[vid_name]["img_feat"] = vid_arr