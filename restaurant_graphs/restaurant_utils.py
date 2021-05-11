from collections import Counter
from itertools import combinations, chain
from operator import itemgetter

import pandas as pd
import numpy as np


def calculate_popularity(df):
    # Calculate popularity value according to formula
    df["popularity_value"] = (df["raw_stars"] * df["review_count"] +
        df["raw_stars"].mean() * (df["tip_count"] + df["checkin_count"]))

    bottom = df["popularity_value"].quantile(0.33)
    top = df["popularity_value"].quantile(0.67)

    def classify_popular(score):
        if score >= top:
            return 2
        if score >= bottom:
            return 1
        return 0

    df["popularity"] = df["popularity_value"].apply(classify_popular)

    return df


def extract_attributes(restaurant_df, business_df):
    def get_attribute_value(att):
        def _get_att(x):
            try:
                if att in x:
                    return int(x[att])
                return 1
            except:
                return 1
        return _get_att

    def get_attribute_bool(att):
        def _get_att(x):
            try:
                if att in x:
                    if x[att]:
                        return 1
                return 0
            except:
                return 0
        return _get_att
    
    
    business_df["RestaurantsPriceRange2"] = business_df["attributes"].apply(get_attribute_value("RestaurantsPriceRange2"))
    
    bool_attributes = [
        "OutdoorSeating", "RestaurantsGoodForGroups", "BusinessAcceptsCreditCards",
        "GoodForKids", "RestaurantsDelivery", "Caters"
    ]
    for att in bool_attributes:
        business_df[att] = business_df["attributes"].apply(get_attribute_bool(att))

    df = restaurant_df.merge(business_df, on="business_id")
    df = df.drop(["attributes_y", "attributes_x"], axis=1)
    return df


def get_top_k_p_combinations(df, comb_p, topk, output_freq=False):
    """
    params:
        df: input dataframe
        comb_p: number of elements in each combination (e.g., there are two
                elements in the combination {fried chicken, chicken and waffle},
                and three elements in the combination {fried chicken, chicken
                and waffle, chicken fried rice})
        topk: number of mostly frequent combinations to retrieve
        output_freq: whether to return the frequencies of retrieved combinations

    return:
        1. output_freq = True: a list X where each element is a tuple containing
                               a combinantion tuple and corresponding frequency,
                               and the elements are stored in the descending
                               order of their frequencies
        2. output_freq = False: a list X where each element is a tuple containing
                                a combinantion tuple, and the elements are stored
                                in the descending order of their frequencies
    """

    # get all combinations with comb_p
    def _get_category_combinations(categories_str):
        if categories_str is None:
            return []
        categories = categories_str.split(', ')
        return list(combinations(categories, comb_p))

    all_categories_p_combos = df["categories"].apply(
        _get_category_combinations).values.tolist()
    # list of tuples that each index refer to one combination
    all_categories_p_combos = list(chain(*all_categories_p_combos))

    category_combo_counter = Counter(all_categories_p_combos)
    sorted_categories_combinations = sorted(category_combo_counter.items(),
                                            key=lambda x: x[1],
                                            reverse=True)
    if output_freq:
        return sorted_categories_combinations[:topk]
    else:
        return [t[0] for t in sorted_categories_combinations[:topk]]


def get_cat2idx(df, topk=500):
    # Collect the categories of all items
    all_categories = [
        category for category_list in df["categories"].values
        for category in category_list.split(", ")
    ]

    # Sort all unique values of the item categories by their frequencies in descending order
    category_sorted = sorted(Counter(all_categories).items(),
                             key=lambda x: x[1],
                             reverse=True)

    # Select top k most frequent categories
    selected_categories = [t[0] for t in category_sorted[:topk]]

    # Create a dictionary mapping each secleted category to a unique integral index
    cat2idx = {cat: idx for idx, cat in enumerate(selected_categories, 1)}

    # Map all categories unseen in the item df to index 0
    cat2idx['unk'] = 0

    # Create a dictionary mapping each integral index to corresponding category
    idx2cat = {idx: cat for cat, idx in cat2idx.items()}

    return cat2idx, idx2cat


def get_category_features(df, cat2idx, top_combo):
    """
    params:
        -df: input dataframe
        -cat2idx: a dictionary mapping item categories to corrresponding integral indices
        -top_combo: a list containing retrieved mostly frequent combinantions of item categories

    return:
        a numpy array where each row contains the categorical features' binary encodings and cross product
        transformations for the corresponding row of the input dataframe
    """

    def _categories_to_binary_output(categories):
        binary_output = np.zeros(len(cat2idx))
        for category in categories.split(", "):
            binary_output[cat2idx.get(category, 0)] = 1
        return binary_output

    def _categories_cross_transformation(categories):
        current_category_set = set(categories.split(", "))
        cross_transform_output = np.zeros(len(top_combo))
        for k, comb_k in enumerate(top_combo):
            comb_k = set(comb_k)
            if comb_k.issubset(current_category_set):
                cross_transform_output[k] = 1
            else:
                cross_transform_output[k] = 0
        return cross_transform_output

    df["top_categories_vector"] = df["categories"].apply(_categories_to_binary_output)
    df["top_categories_combination_vector"] = df["categories"].apply(_categories_cross_transformation)

    return df


def get_encoder(df):
    encoder = {}
    for col_name in ["city", "state"]:
        encoder[col_name] = {
            item: idx for idx, item in enumerate(df[col_name].unique(), 1)
        }
    return encoder

def encode_cities_states(df, encoder):
    for col_name in ["city", "state"]:
        df[col_name + "_idx"] = df[col_name].apply(lambda x: encoder[col_name].get(x, 0))

    return df


def to_categorical(y, num_classes=None, dtype='float32'):
    """Direct fork from keras.utils.to_categorical
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical