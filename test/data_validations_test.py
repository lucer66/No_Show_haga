import pandas as pd
import pytest
from noshow.preprocessing.load_data import (
    load_appointment_csv,
    process_appointments,
    process_postal_codes,
)
from noshow.config import CLINIC_CONFIG
from noshow.features.feature_pipeline import create_features
from noshow.preprocessing.geo import (
    haversine_distance,
)

@pytest.fixture
def df():
    appointments_df = load_appointment_csv("./data/raw/poliafspraken_no_show.csv", 10)
    appointments_df = process_appointments(appointments_df, CLINIC_CONFIG)
    all_postalcodes = process_postal_codes("./data/raw/NL.csv")
    appointments_features = create_features(appointments_df, all_postalcodes)

    return appointments_features


def test_hour_calculation(df):
    assert df['hour'].notna().all(), "Calculated 'hour' has missing values"
    assert df['hour'].between(0, 23).all(), "'hour' values are out of range (0-23)"
    assert (df['hour'] == df.index.get_level_values('start').hour).all(), "Calculated 'hour' does not match the hour part of 'start'"

def test_weekday_calculation(df):
    assert df['weekday'].notna().all(), "Calculated 'weekday' has missing values"
    assert df['weekday'].between(0, 6).all(), "'weekday' values are out of range (0-6)"
    assert (df['weekday'] == df.index.get_level_values('start').weekday).all(), "Calculated 'weekday' does not match the weekday part of 'start'"


def test_no_show_calculation(df):
    assert df['no_show'].isin(['show', 'no_show']).all(), "'no_show' has invalid values"
    
    # Check if 'no_show' equals 'no_show' when cancellationReason_code equals 'N' or 'NF'
    no_show_condition = df['cancelationReason_code'].isin(['N', 'NF'])
    assert (df.loc[no_show_condition, 'no_show'] == 'no_show').all(), "'no_show' is not 'no_show' when cancelationReason_code is 'N' or 'NF'"



def test_previous_no_show_percentage(df):
    assert df['prev_no_show_perc'].between(0, 100).all(), "'prev_no_show_perc' is out of range"
    
    # Validate the calculation of 'prev_no_show_perc'
    for _, row in df.iterrows():
        if row['earlier_appointments'] > 0:
            expected_perc = row['prev_no_show'] / row['earlier_appointments'] * 100
            assert abs(row['prev_no_show_perc'] - expected_perc) < 1e-6, \
                f"Expected 'prev_no_show_perc' for index {row.name} to be {expected_perc}, but got {row['prev_no_show_perc']}"
        else:
            assert row['prev_no_show_perc'] == 0, \
                f"Expected 'prev_no_show_perc' to be 0 for index {row.name} with 0 earlier appointments"


def test_add_days_since_created(df):
    assert df['days_since_created'].notna().all(), "'days_since_created' has missing values"
    assert (df['days_since_created'] >= 0).all(), "'days_since_created' has negative values"
    assert (df.loc[df['days_since_created'] < 0, 'days_since_created'] == 0).all(), "'days_since_created' is not set to 0 where expected"


def test_add_days_since_last_appointment(df):
    assert df['days_since_last_appointment'].notna().all(), "'days_since_last_appointment' has missing values"
    assert (df['days_since_last_appointment'] >= 0).all(), "'days_since_last_appointment' has negative values"
    
    for _, row in df.iterrows():
        pseudo_id = row.name[0]  
        if pseudo_id != row.name[0]:
            previous_row = df.loc[(df.index.get_level_values('pseudo_id') == pseudo_id) & (df.index.get_level_values('start') < row['start'])].iloc[-1]
            expected_days = (row['start'] - previous_row['start']).days
            assert row['days_since_last_appointment'] == expected_days, \
                f"Expected 'days_since_last_appointment' for pseudo_id {pseudo_id} to be {expected_days}, but got {row['days_since_last_appointment']}"


def test_add_appointments_last_days(df):
    assert df['appointments_last_days'].notna().all(), "'appointments_last_days' has missing values"
    assert (df['appointments_last_days'] >= 0).all(), "'appointments_last_days' has negative values"
    
    for _, row in df.iterrows():
        pseudo_id = row.name[0]
        if pseudo_id != row.name[0]: 
            count_in_last_days = df.loc[
                (df.index.get_level_values('pseudo_id') == pseudo_id) &
                (df.index.get_level_values('start') >= row['start'] - pd.Timedelta(days=14))
            ]['APP_ID'].count()
            assert row['appointments_last_days'] == count_in_last_days, \
                f"Expected 'appointments_last_days' for pseudo_id {pseudo_id} to be {count_in_last_days}, but got {row['appointments_last_days']}"

def test_add_appointments_same_day(df):
    assert df['appointments_same_day'].notna().all(), "'appointments_same_day' has missing values"
    
    for _, row in df.iterrows():
        pseudo_id = row.name[0]
        start_date = row.name[1].date()

        appointments_same_day_count = df.loc[
            (df.index.get_level_values('pseudo_id') == pseudo_id) & 
            (df.index.get_level_values('start').date == start_date)
        ].shape[0]
        
        assert row['appointments_same_day'] == appointments_same_day_count, \
            f"Expected 'appointments_same_day' for pseudo_id {pseudo_id} and date {start_date} to be {appointments_same_day_count}, but got {row['appointments_same_day']}"


def test_add_minutes_early(df):
    assert df['minutes_early'].notna().all(), "'minutes_early' has missing values"
    
    assert (df['minutes_early'] >= -60).all(), "'minutes_early' has values below -60"
    assert (df['minutes_early'] <= 60).all(), "'minutes_early' has values above 60"
    
    assert df['prev_minutes_early'].notna().all(), "'prev_minutes_early' has missing values"
    
    for _, row in df.iterrows():
        pseudo_id = row.name[0]  # Access pseudo_id from the index
        if row['earlier_appointments'] > 0:
            expected_prev_minutes_early = (df.loc[
                (df.index.get_level_values('pseudo_id') == pseudo_id) & 
                (df.index.get_level_values('start') < row['start'])
            ]['minutes_early'].sum()) / row['earlier_appointments']
            assert abs(row['prev_minutes_early'] - expected_prev_minutes_early) < 1e-6, \
                f"Expected 'prev_minutes_early' for pseudo_id {pseudo_id} to be {expected_prev_minutes_early}, but got {row['prev_minutes_early']}"
        else:
            assert row['prev_minutes_early'] == 0, "'prev_minutes_early' should be 0 when there are no earlier appointments"

def test_add_dist_umcu(df):
    assert df['dist_umcu'].notna().all(), "'dist_umcu' has missing values"
    assert pd.api.types.is_numeric_dtype(df['dist_umcu']), "'dist_umcu' is not numeric"
    assert (df['dist_umcu'] >= 0).all(), "'dist_umcu' has negative values"
    
    # Manually calculate the expected distance using the haversine function for each row
    for _, row in df.iterrows():
        # Access latitude and longitude from the DataFrame
        latitude = float(row['latitude'])
        longitude = float(row['longitude'])
        
        hospital_name = "HagaZiekenhuis Den Haag" 
        
        # Calculate the expected distance using haversine_distance
        expected_distance = haversine_distance(latitude, longitude, hospital_name)
        
        assert abs(row['dist_umcu'] - expected_distance) < 1e-6, \
            f"Expected 'dist_umcu' for pseudo_id {row.name[0]} to be {expected_distance}, but got {row['dist_umcu']}"




def run_data_validation(df):
    test_missing_values(df)
    test_duplicates(df)
    test_feature_ranges(df)
    test_categorical_encoding(df)
    test_filtering_correctness(df)
    print("All data validation checks passed!")
