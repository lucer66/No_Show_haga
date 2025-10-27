import numpy as np

from math import atan2, cos, radians, sin, sqrt


def haversine_distance(
    lat1: float,
    lon1: float,
    description: str,
) -> float:
    """Calculate Haversine distance

    Default second location is the location of the HagaZiekenhuis

    Parameters
    ----------
    lat1 : float
        Latitude of location 1
    lon1 : float
        Longitude of location 1
    description : string, 
        string with the location of the hospital
    Returns
    -------
    float
        The distance between both points in kilometers
    """
    #if (np.isnan(lat1)): return 5.0


    HospitalCords = {
        "HagaZiekenhuis Den Haag": {"lat": 52.055390188060734, "lon": 4.263833234196031},
        "HagaZiekenhuis Zoetermeer": {"lat": 52.068876630412916, "lon": 4.499605937876778},
        "Reinier Haga Orthopedisch Centrum (Locatie: Zoetermeer)": {"lat": 52.068876630412916, "lon": 4.499605937876778},
        "Juliana kinderziekenhuis": {"lat": 52.05537452123235, "lon": 4.264036117637258},
        "Bijzondere tandheelkunde, locatie Leyweg": {"lat": 52.055390188060734, "lon": 4.263833234196031},
        "HagaZiekenhuis Europaweg Zoetermeer": {"lat": 52.06370632002677, "lon": 4.494390756080904},
        "Reinier Haga Orthopedisch Centrum (Locatie: Den Haag)": {"lat": 52.055390188060734, "lon": 4.263833234196031},
        "Reinier Haga Orthopedisch Centrum (Locatie: Reinier de Graaf Gasthuis)": {"lat": 51.997912460653126, "lon": 4.339419973455814},
        "HagaZiekenhuis Huidkliniek Wassenaar": {"lat": 52.14661737707496, "lon": 4.399405530507219},
    }

    coords = HospitalCords.get(description)
    if (coords == None): coords = HospitalCords.get("HagaZiekenhuis Den Haag") #default, most likely location

    lat2 = coords["lat"]
    lon2 = coords["lon"]
    
        
    R = 6373.0  # approximate radius of Earth in km

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance
