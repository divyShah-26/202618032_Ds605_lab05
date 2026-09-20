import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split



BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


model = joblib.load(MODEL_DIR / "airbnb_xgb_118.pkl")

preprocessor = joblib.load(
    MODEL_DIR / "airbnb_preprocessor.pkl"
)

tfidf = joblib.load(
    MODEL_DIR / "airbnb_tfidf_100.pkl"
)

interaction_te = joblib.load(
    MODEL_DIR / "neighbourhood_room_type_te.pkl"
)

te_global_mean = joblib.load(
    MODEL_DIR / "te_global_mean.pkl"
)



@st.cache_resource
def create_neighbourhood_encoding():

    data_path = DATA_DIR / "AB_NYC_2019.csv"

    df = pd.read_csv(data_path)

    # Same filtering used in the notebook
    df = df[df["price"] > 0].copy()

    # Same 99th percentile price cap
    cap = df["price"].quantile(0.99)

    df["price_capped"] = df["price"].clip(upper=cap)

    # Features used for the final model
    features = [
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365"
    ]

    X = df[features].copy()
    y = df["price_capped"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    y_train_log = np.log1p(y_train)

    train_enc = X_train.copy()
    train_enc["log_price"] = y_train_log.values

    neighbourhood_target_mean = (
        train_enc
        .groupby("neighbourhood")["log_price"]
        .mean()
    )

    global_mean = y_train_log.mean()

    return neighbourhood_target_mean, global_mean, cap


neighbourhood_te, reconstructed_global_mean, PRICE_CAP = (
    create_neighbourhood_encoding()
)


st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)


st.title("🏠 NYC Airbnb Price Predictor")

st.write(
    "Enter the details of an Airbnb listing to estimate "
    "its nightly price."
)

st.divider()



@st.cache_data
def load_neighbourhoods():

    data_path = DATA_DIR / "AB_NYC_2019.csv"

    data = pd.read_csv(
        data_path,
        usecols=["neighbourhood"]
    )

    return sorted(
        data["neighbourhood"]
        .dropna()
        .unique()
        .tolist()
    )


neighbourhoods = load_neighbourhoods()



neighbourhood = st.selectbox(
    "Neighbourhood",
    neighbourhoods
)


col1, col2 = st.columns(2)


with col1:

    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        [
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Bronx",
            "Staten Island"
        ]
    )

    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )

    latitude = st.number_input(
        "Latitude",
        min_value=40.49,
        max_value=40.92,
        value=40.7580,
        format="%.6f"
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-74.25,
        max_value=-73.70,
        value=-73.9855,
        format="%.6f"
    )

    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        max_value=365,
        value=3
    )



with col2:

    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        max_value=1000,
        value=50
    )

    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.1
    )

    calculated_host_listings_count = st.number_input(
        "Host Listing Count",
        min_value=1,
        max_value=400,
        value=1
    )

    availability_365 = st.number_input(
        "Availability (days/year)",
        min_value=0,
        max_value=365,
        value=200
    )

    name = st.text_input(
        "Listing Name",
        value="Beautiful cozy apartment near Central Park"
    )




st.divider()


if st.button(
    "Predict Nightly Price",
    type="primary"
):

    

    input_df = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count":
            calculated_host_listings_count,
        "availability_365": availability_365
    }])


    
    input_df["reviews_per_month"] = (
        input_df["reviews_per_month"]
        .fillna(0)
    )


    

    manhattan_lat = 40.7580
    manhattan_lon = -73.9855

    input_df["distance_to_manhattan"] = np.sqrt(
        (input_df["latitude"] - manhattan_lat) ** 2
        +
        (input_df["longitude"] - manhattan_lon) ** 2
    )


    

    input_df["neighbourhood_te"] = (
        input_df["neighbourhood"]
        .map(neighbourhood_te)
        .fillna(reconstructed_global_mean)
    )


    
    input_df["neighbourhood_room_type"] = (
        input_df["neighbourhood"].astype(str)
        + "__"
        + input_df["room_type"].astype(str)
    )


   

    interaction_value = (
        input_df["neighbourhood_room_type"]
        .map(interaction_te)
        .fillna(te_global_mean)
        .iloc[0]
    )


   

    structured_input = input_df.drop(
        columns=[
            "neighbourhood",
            "neighbourhood_room_type"
        ]
    )


   
    

    structured_features = preprocessor.transform(
        structured_input
    )


   

    name_features = tfidf.transform([name])


    final_input = hstack(
        [
            csr_matrix(structured_features),
            name_features,
            csr_matrix([[interaction_value]])
        ],
        format="csr"
    )


    

    prediction = model.predict(
        final_input
    )[0]



    prediction = np.clip(
        prediction,
        0,
        PRICE_CAP
    )


   

    st.success(
        f"Estimated nightly price: **${prediction:.2f}**"
    )

    st.caption(
        "Prediction generated using the trained XGBoost model."
    )