from pathlib import Path

import pandas as pd

from noshow.config import APPOINTMENTS_LAST_DAYS, CLINIC_CONFIG, MINUTES_EARLY_CUTOFF
from noshow.features.appointment_features import (
    add_appointments_last_days,
    add_appointments_same_day,
    add_days_since_created,
    add_days_since_last_appointment,
    add_minutes_early,
    add_time_features,
)
from noshow.features.no_show_features import prev_no_show_features
from noshow.features.patient_features import add_patient_features
from noshow.preprocessing.load_data import (
    load_appointment_csv,
    process_appointments,
    process_postal_codes,
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

def create_features(
    appointments_df: pd.DataFrame,
    all_postal_codes: pd.DataFrame,
) -> pd.DataFrame:
    """Create all the feature for the no-show model

    This function is a pipeline that applies all the feature
    creation code to generate a feature table.

    Parameters
    ----------
    appointments_df : pd.DataFrame
        The dataframe containing all the appointment data, see the
        process_appointments function on how to read this data.
    all_postal_codes : pd.DataFrame
        The dataframe containing all the postalcodes in the
        Netherlands

    Returns
    -------
    pd.DataFrame
        The featuretable
    """
    appointments_features = (
        appointments_df.pipe(prev_no_show_features)
        .pipe(add_appointments_same_day)
        .pipe(add_days_since_last_appointment)
        .pipe(add_days_since_created)
        .pipe(add_appointments_last_days, APPOINTMENTS_LAST_DAYS)
        .pipe(add_minutes_early, MINUTES_EARLY_CUTOFF)
        .pipe(add_time_features)
        .pipe(add_patient_features, all_postal_codes)
        .sort_index(level="start")
    )

    return appointments_features



def select_feature_columns(featuretable: pd.DataFrame) -> pd.DataFrame:
    return featuretable[
        [
            "hour",
            "weekday",
            "minutesDuration",
            "no_show",
            "prev_no_show",
            "prev_no_show_perc",
            "age",
            "dist_umcu",
            "gender",
            "prev_minutes_early",
            "earlier_appointments",
            "appointments_same_day",
            "appointments_last_days",
            "days_since_created",
            "days_since_last_appointment",
        ]
    ]

#scales all features if needed for a different model
def log_filters(featuretable: pd.DataFrame) -> pd.DataFrame:
    categorical = ['gender', 'hour', 'weekday',]
    unscaled = [ 'no_show', 'prev_no_show_perc', 'age']
    numerical = ['no_show', "prev_no_show","prev_no_show_perc","age","dist_umcu","prev_minutes_early","earlier_appointments","appointments_same_day","appointments_last_days","days_since_created","days_since_last_appointment"]

    scaled_numnerical = [col for col in numerical if col not in unscaled]

    processor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), scaled_numnerical),
            ('cat', OneHotEncoder(drop="first"), categorical)
        ],
        remainder='passthrough'
    )

    transformed = processor.fit_transform(featuretable)
    new_cols = processor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=new_cols, index=featuretable.index)
    transformed_df = transformed_df.rename(columns={'remainder__no_show': 'no_show', 'remainder__prev_no_show_perc': 'prev_no_show_perc'})
   
    return transformed_df

#Preprocessing specifically for XGBoost
def preprocess_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    categorical = ['gender', 'hour', 'weekday',]

    df_cat = df[categorical].astype("category")
    df_num = df.drop(columns=categorical)
    df_cat_encoded = pd.get_dummies(df_cat, drop_first=True)

    df_processed = pd.concat([df_num.reset_index(drop=True), df_cat_encoded.reset_index(drop=True)], axis =1)
    
    return df_processed


if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "raw"
    output_path = Path(__file__).parents[3] / "data" / "processed"
    appointments_df = load_appointment_csv(data_path / "poliafspraken_no_show.csv")
    appointments_df = process_appointments(appointments_df, CLINIC_CONFIG)
    all_postalcodes = process_postal_codes(data_path / "NL.csv")

    appointments_features = (
        create_features(appointments_df, all_postalcodes)
        .pipe(select_feature_columns)
        #.pipe(log_filters) #uncomment when using a specific model
        #.pipe(preprocess_xgboost) #uncomment when using a specific model
        .pipe(test)
        .to_parquet(output_path / "featuretable.parquet")
    )

