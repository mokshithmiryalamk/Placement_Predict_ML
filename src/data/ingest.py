import os #Imports Python's built-in Operating System module
import pandas as pd #Imports the Pandas library

def load_and_validate_data(file_path: str) -> pd.DataFrame: #Defines a reusable function
    #Loads CSV data and validates schema integrity.

    if not os.path.exists(file_path): #Checks whether file exists.
        raise FileNotFoundError(
            f"Critical Error: Targeted data footprint not discovered at {file_path}"
        ) #Stops execution immediately.

    print(f"Executing secure data extraction from: {file_path}") #Displays progress message

    df = pd.read_csv(file_path) #Loads CSV into memory

    required_columns = [
    'branch',
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count',
    'placement_status',
    'salary_package_lpa'
] #These are the features needed for training.

    missing_cols = [col for col in required_columns if col not in df.columns] #Find missing columns

    if missing_cols: #Checks whether list is non-empty
        raise ValueError(
            f"Schema Validation Failure: Missing essential feature targets: {missing_cols}"
        ) #Stops processing if schema is incorrect

    print(
        f"Data ingestion resolved successfully. "
        f"Dimensions captured: {df.shape[0]} samples, {df.shape[1]} metrics." #Displays dataset size
    )

    return df #Returns DataFrame to caller


if __name__ == "__main__": #Executes code only when file is run directly

    DATA_PATH = os.path.join("src", "data", "raw_placement_data.csv") #Creates OS-independent file path

    try: #Begin exception handling
        raw_data = load_and_validate_data(DATA_PATH)

    except Exception as e: #Catch any unexpected error
        print(f"Ingestion lifecycle termination: {str(e)}") #Display human-readable error message