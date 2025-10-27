import pandas as pd
from datetime import datetime

from noshow.preprocessing.geo import haversine_distance


def add_patient_features(
    appointments_df: pd.DataFrame, all_postalcodes: pd.DataFrame
) -> pd.DataFrame:
    """Add patient age and distance to UMCU

    Parameters
    ----------
    appointments_df : pd.DataFrame
        A dataframe containing info on appointments,
        needs to contain the `address_postalCodeNumbersNL` and `BIRTH_YEAR`
        columns.
    all_postalcodes : pd.DataFrame
        A dataframe containing all postalcodes in the Netherlands with location.
        Needs to have a index on postalcode and columns `longitude` and `latitude`.

    Returns
    -------
    pd.DataFrame
        The appointment_df dataframe with added columns `age` and `dist_umcu`.
    """

    appointments_df["address_postalCodeNumbersNL"] = appointments_df["address_postalCodeNumbersNL"].astype(str)
    all_postalcodes.index = all_postalcodes.index.astype(str)

    appointments_df = appointments_df.merge(
        all_postalcodes, left_on="address_postalCodeNumbersNL", right_index=True, how="left"
    )

    #filter invalid latitudes
    appointments_df = appointments_df.loc[
        ~appointments_df["latitude"].isna()
    ]
    appointments_df = appointments_df.loc[
        ~appointments_df["longitude"].isna()
    ]

    #dist haga
    appointments_df["dist_umcu"] = appointments_df.apply(
        lambda x: haversine_distance(float(x["latitude"]), float(x["longitude"]), "Hagaziekenhuis Den Haag"), axis="columns"
    )
    
    appointments_df["age"] = (
        datetime.now().year - appointments_df["BIRTH_YEAR"]
    )
    return appointments_df