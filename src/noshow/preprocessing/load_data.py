from datetime import date
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import numpy as np

from noshow.api.pydantic_models import Appointment
from noshow.config import ClinicConfig


def load_appointment_pydantic(input: List[Appointment]) -> pd.DataFrame:
    """Load prediction data from a list of Appointments

    Parameters
    ----------
    input : List[Appointment]
        The input data as a list of Appointments

    Returns
    -------
    pd.DataFrame
        The loaded data as pandas dataframe
    """
    appointments_df = pd.DataFrame([a.model_dump() for a in input])
    appointments_df = appointments_df.replace("", None)

    # Clinic name and description are sometimes unknown since HiX6.3
    appointments_df.loc[appointments_df["name"].isnull(), "name"] = "Onbekend"
    appointments_df.loc[appointments_df["description"].isnull(), "description"] = (
        "Onbekend"
    )
    return appointments_df


def load_appointment_csv(csv_path: Union[str, Path], nrows = None) -> pd.DataFrame:
    """Load data from a csv file

    The query used to create this CSV can be found in the data folder

    Parameters
    ----------
    csv_path : Union[str, Path]
        The path to the csv file

    Returns
    -------
    pd.DataFrame
        The pandas dataframe from the csv file
    """
        
    appointments_df = pd.read_csv(
        csv_path,
        nrows=nrows,
        sep=',',
        header=None,
        names=["APP_ID", "pseudo_id", "wat_id?", "hoofdagenda", "hoofdagenda_id", "subagenda_id", "specialty_code", "soort_consult", "afspraak_code", "start", "end", "arrival", "created", "minutesDuration", "status", "status_code_original", "cancelationReason_code", "cancelationReason_display", "BIRTH_YEAR", "address_postalCodeNumbersNL", "name", "description", "ziekenhuis", "soort_cons", "ConsultTypeDescription", "geen_idee", "gender", "geen_id"],
        encoding='utf-8',
        parse_dates=["created"],
        date_format="ISO8601",
        dtype={"specialty_code": "object", "status": "object", "cancelationReason_code": "object"},
    )

    appointments_df["start"] = pd.to_datetime(
        appointments_df["start"], errors="coerce", format="ISO8601"
    )
    appointments_df["end"] = pd.to_datetime(
        appointments_df["end"], errors="coerce", format="ISO8601"
    )
    appointments_df["arrival"] = pd.to_datetime(
        appointments_df["arrival"], errors="coerce", format="ISO8601"
    )

    return appointments_df


def process_appointments(
    appointments_df: pd.DataFrame,
    clinic_config: Dict[str, ClinicConfig],
    start_date: str | None = None,
) -> pd.DataFrame:
    """Process the appointments data

    Parameters
    ----------
    appointments_df : Union[str, Path]
        The pandas dataframe with appointments data from either csv or json
    clinic_config : Dict[str, ClinicConfig]
        The clinic configuration, containing filters and clinic info
    start_date : str, optional
        The start date for the predictions, if given will filter out appointments of
        patients that do not have an appointment on this date, by default None

    Returns
    -------
    pd.DataFrame
        Cleaned appointment DataFrame that can be used for feature building
    """
    appointments_df = apply_config_filters(appointments_df, clinic_config, start_date)

    appointments_df["no_show"] = "show"

    #NOTE CHECK THIS LOGIC
    appointments_df.loc[appointments_df["cancelationReason_code"].isin(["N", "NF"]), "no_show"] = "no_show"

    # Some patients have multiple postal codes
    appointments_df = appointments_df.drop_duplicates(
        subset=appointments_df.columns.difference(["address_postalCodeNumbersNL"])
    )

    # Some start dates are NaT
    print('before start dates NaT onbekend', len(appointments_df))
    appointments_df = appointments_df.loc[~appointments_df["start"].isna()]

    #Filter out the covid years
    appointments_df['start'] = pd.to_datetime(appointments_df['start'], unit='s')
    appointments_df = appointments_df[appointments_df['start'] >= '2021-06-01']
    appointments_df = appointments_df.loc[~appointments_df["address_postalCodeNumbersNL"].isna()]

    #gender in binary
    appointments_df["gender"] = (
        appointments_df["gender"].replace({"Man": "1", "Vrouw": "0"}).astype(int)
    )

    #transform is voldaan / is niet voldaan to 1 / 0
    appointments_df["status"] = (
        appointments_df["status"].replace({"Is voldaan": "1", "Is niet voldaan": "0"}).astype(int)
    )

    appointments_df = appointments_df.set_index(["pseudo_id", "start"])

    # Rolling features can't be calculated on non-unique index
    appointments_df = appointments_df[~appointments_df.index.duplicated(keep="last")]

    return appointments_df


def process_postal_codes(postalcodes_path: Union[str, Path]) -> pd.DataFrame:
    """Load and process all postalcode locations in the Netherlands

    This file can be found at: https://download.geonames.org/export/zip/NL.zip

    Parameters
    ----------
    postalcodes_path : Union[str, Path]
        Path to the tsv-file that contains postalcode information

    Returns
    -------
    pd.DataFrame
        A dataframe containing all postalcodes and longlat positions
        in the Netherlands
    """
    all_postalcodes = pd.read_table(
        postalcodes_path,
        sep=",",
        header=None,
        names=[
            "country",
            "postalcode",
            "city",
            "admin_name1",
            "admin_code1",
            "admin_name2",
            "admin_code2",
            "admin_name3",
            "admin_code3",
            "latitude",
            "longitude",
            "accuracy",
        ],
    )
    all_postalcodes = all_postalcodes.set_index("postalcode")[["latitude", "longitude"]]
    all_postalcodes = all_postalcodes.loc[~all_postalcodes.index.duplicated()]
    return all_postalcodes


def apply_config_filters(
    appointments_df: pd.DataFrame,
    clinic_config: Dict[str, ClinicConfig],
    start_date: str | None = None,
) -> pd.DataFrame:
    """Apply the clinic config filters to the appointments data

    Parameters
    ----------
    appointments_df : pd.DataFrame
        The appointments data
    clinic_config : Dict[str, ClinicConfig]
        The clinic configuration, containing filters and clinic info
    start_date : str, optional
        The start date for the predictions, if given will filter out appointments of
        patients that do not have an appointment on this date, by default None

    Returns
    -------
    pd.DataFrame
        The appointments data with the clinic config filters applied
    """
    clinic_df_list = []
    for name, config in clinic_config.items():
        clinic_df = appointments_df
        #clinic_df = appointments_df.loc[
        #    appointments_df["hoofdagenda_id"].isin(config.main_agenda_codes)
        #].copy()
        #clinic_df["clinic"] = name

        # if config.subagenda_exclude and config.subagendas:
        #     clinic_df = clinic_df.loc[
        #         ~clinic_df["subagenda_id"].isin(config.subagendas)
        #     ]
        # elif config.subagendas:
        #     clinic_df = clinic_df.loc[clinic_df["subagenda_id"].isin(config.subagendas)]

        # if config.appcode_exclude and config.appcodes:
        #     clinic_df = clinic_df.loc[~clinic_df["afspraak_code"].isin(config.appcodes)]
        # elif config.appcodes:
        #     clinic_df = clinic_df.loc[clinic_df["afspraak_code"].isin(config.appcodes)]
        

        #Filtering out certain clinics
        clinic_df = clinic_df.loc[clinic_df["hoofdagenda"].isin(
            ["OOGHEELKUNDE"]
        )]
    
        #Filtering out locations
        clinic_df = clinic_df.loc[appointments_df["ziekenhuis"].isin(
            ["HagaZiekenhuis Den Haag"] #, "HagaZiekenhuis Zoetermeer"
        )]

        
        # No phone consults
        clinic_df = clinic_df.loc[
            clinic_df["soort_consult"] != "Telefonisch"
        ]
    
        #filter out invalid status cancellation combination
        clinic_df = clinic_df.loc[
            ~((clinic_df["status"] == "Is niet voldaan") & (clinic_df["cancelationReason_code"].isin(["N", "NF"])))
        ]
    
        #filter out invalid OOG clinics for modelling
        clinic_df = clinic_df[~clinic_df['afspraak_code'].isin(['VNAVAST', 'VNEYLEA'])]
        

        # filter out patients with status 'Is niet voldaan' and cancelationReasoncode NULL
        clinic_df = clinic_df.loc[
            ~((clinic_df["status"] == "Onbekend"))
        ]

        clinic_df = clinic_df.loc[
            ~((clinic_df["gender"] == "Onbekend"))
        ]

        clinic_df = clinic_df.loc[
            ~((clinic_df["address_postalCodeNumbersNL"] == ""))
        ]

        #filter out found placeholder postal codes
        clinic_df = clinic_df.loc[
            ~((clinic_df["address_postalCodeNumbersNL"] == "9999"))
        ]

        clinic_df = clinic_df.loc[
            ~((clinic_df["address_postalCodeNumbersNL"] == "0000"))
        ]

        clinic_df_list.append(clinic_df)

    total_df = pd.concat(clinic_df_list)

    # After filtering there could still be appointments of patients that no longer have
    # an appointment on the start date.
    if start_date:
        patient_list = total_df.loc[
            total_df["start"].dt.date == date.fromisoformat(start_date), "pseudo_id"
        ].unique()
        total_df = total_df.loc[total_df["pseudo_id"].isin(patient_list)]

    return total_df
