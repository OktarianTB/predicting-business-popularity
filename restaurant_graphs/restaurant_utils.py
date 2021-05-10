import pandas as pd
import numpy as np
from operator import itemgetter


def calculate_popularity(df):
    # Calculate popularity value according to formula
    df["popularity_value"] = df["raw_stars"] * df["review_count"] + \
        df["raw_stars"].mean() * (df["tip_count"] + df["checkin_count"])

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
    def get_attribute_value(att, x):
        try:
            if att in x:
                return int(x[att])
            return 1
        except:
            return 1

    def get_attribute_bool(att, x):
        try:
            if att in x:
                if x[att]:
                    return 1
            return 0
        except:
            return 0

    business_df["price_tier"] = business_df["attributes"].apply(
        lambda x: get_attribute_value("RestaurantsPriceRange2", x))

    business_df["outdoors"] = business_df["attributes"].apply(
        lambda x: get_attribute_bool("OutdoorSeating", x))
    business_df["good_for_groups"] = business_df["attributes"].apply(
        lambda x: get_attribute_bool("RestaurantsGoodForGroups", x))
    business_df["has_credit_card"] = business_df["attributes"].apply(
        lambda x: get_attribute_bool("BusinessAcceptsCreditCards", x))
    business_df["good_for_kids"] = business_df["attributes"].apply(
        lambda x: get_attribute_bool("GoodForKids", x))
    business_df["has_delivery"] = business_df["attributes"].apply(
        lambda x: get_attribute_bool("RestaurantsDelivery", x))
    business_df["caters"] = business_df["attributes"].apply(
        lambda x: get_attribute_bool("Caters", x))

    df = restaurant_df.merge(business_df, on="business_id")
    df.drop(["attributes_y", "attributes_x"], axis=1)
    return df


def get_top_categories(df, n):
    categories = {}

    for _, row in df.iterrows():
        cats = [cat.lower().strip() for cat in row["categories"].split(",")]
        for c in cats:
            if c in categories.keys():
                categories[c] += 1
            else:
                categories[c] = 1

    categories_list = list(categories.items())
    categories_list = sorted(categories_list, key=itemgetter(1), reverse=True)
    top_categories = [name for (name, count) in categories_list[:n]]
    return top_categories


def get_categories(categories, top_categories):
    categories = categories.lower()
    categories_list = []
    for top_cat in top_categories:
        if top_cat in categories:
            categories_list.append(1)
        else:
            categories_list.append(0)
    return categories_list
