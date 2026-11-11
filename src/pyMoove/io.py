"""
Simply load/save data, assumption is that the structure of data after cleaning/modification from AniSpace/moo4r retains initial structure

Modularized and generalized workflow
"""
import gzip
import json
import pandas as pd
from pathlib import Path

#NOTE: Potential issue: 24h data-set may be fine to load completely into RAM, what if data-set spans several days or something? 
#NOTE: Assumptions: Just load whatever data is available, it might be or might not be cleaned right now, decide at a later point
#NOTE: Combining and standardizing columm names should be done separately

#NOTE: Position data IO
#NOTE: Initial profiling shows that loading times are dominated by the individual loading of the files
#NOTE: Initial profiling shows in addition that the inividual file loading is dominated by conversion to pd.DataFrame
#NOTE: Ideas to speed up the process: Try a few approaches of converting to pd.DataFrame, 
#NOTE: Parquet could be used to, worked well for me in the bachelors theis, this is another filetype that is faster
#NOTE: End goal: Individual and unified DataFrames, which can be used to look at at handle stuff.

#TODO: Generalize this dataset loading function to ideally accept 'any' path to a full dataset 
#TODO: Saving functionality once transformed, loading is slow, idea parquet files have worked for me before.
#TODO: -> Saving + some additional loading functionality, once we have the parquet format
#NOTE: Issues may arrise once the dataset grows beyond a single day, considerations to filesize must be made


def load_positions_file(filePath: str | Path) -> pd.DataFrame:
    """
    Load a compressed Pozyx JSON file into a DataFrame
    Assumption: 
        Handles the coordinates subdictionary explicitly, 
        other nested dictionaries are not handled. Reconsider apporach if this changes.
        This approach is about 4x faster than using pd.json_normalize() directly.
        Additional nested dictionaries are therefore preserved and not flattened.
    """

    with gzip.open(filePath, "rt", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    coords = pd.DataFrame(df.pop("coordinates").tolist())
    coords = coords.add_prefix("coordinates.")

    return pd.concat([df, coords], axis=1)


def load_positions_folder(folderPath: str | Path)->pd.DataFrame:
    """Loads all .gz file from a folder
    Assumption: The folder is not empty and contains at least one json.gz"""

    folder = Path(folderPath)

    pandaDataFrames = [
        load_positions_file(filePath=filePath)
        for filePath in sorted(folder.glob("*.json.gz"))
    ]

    fullFrame = pd.concat(pandaDataFrames, ignore_index=True)

    return fullFrame


def load_delpro_tag_mapping(filePath: str | Path)->pd.DataFrame:
    """Loads xlsx file for delpro tags mapping"""
    return pd.read_excel(filePath)


def load_feeding_bin_visits(filePath: str | Path)->pd.DataFrame:
    """Loads feeding bin visits data
    sheet_name="Per visit", since the excel sheet comes in two forms, per visit and per day"""
    return pd.read_excel(filePath, sheet_name="Per visit")


def load_zone_info(filePath: str | Path)->pd.DataFrame:
    """Loads zone_info.csv file
    column separator used here ';', initially"""
    return pd.read_csv(filePath, sep=";")


def load_dataset(dataFolderPath: str | Path) -> dict[str, pd.DataFrame]:
    """Load the positioning, feeding, tag mapping and zone datasets."""
    #NOTE: Hardcoded file and folder names based on initial data,
    #TODO: Generalize this function to ideally accept 'any' path to a full dataset 

    dataFolderPath = Path(dataFolderPath)

    datasetDict = {
        "positions": load_positions_folder(dataFolderPath / "RTLS 20251010"),
        "mapping": load_delpro_tag_mapping(dataFolderPath / "Delpro taggar 20251010.xlsx"),
        "feeding": load_feeding_bin_visits(dataFolderPath / "Utfodring 20251010.xlsx"),
        "zones": load_zone_info(dataFolderPath / "zone_info.csv"),
    }

    return datasetDict
