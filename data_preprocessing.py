import pandas as pd
import json

loan_data = "data/accepted_2007_to_2018Q4.csv.gz"
unemployment_rate_data = ["data/unemployment_rate_0.csv", "data/unemployment_rate_1.csv", "data/unemployment_rate_2.csv", "data/unemployment_rate_3.csv", "data/unemployment_rate_4.csv"]

def preprocess_data(loan_data, unemployment_rate_data):
    """
    Preprocess loan data and merge with unemployment rate data.
    
    Args:
        loan_data (str): Path to the loan data CSV file.
        unemployment_rate_data (list): List of paths to unemployment rate CSV files.

    Returns:
        pd.DataFrame: Preprocessed and merged DataFrame.
    """

    # Load loan data
    loan_data = pd.read_csv(loan_data, low_memory=False)

    # Drop unnecessary id columns
    loan_data = loan_data.drop(columns=['id', 'member_id'])

    # Drop null values in issue_d and loan_status columns
    loan_data = loan_data[loan_data[['issue_d', 'loan_status']].notnull().all(axis=1)]

    # Drop columns with more than 50% null values
    null_percent = (loan_data.isnull().sum() / len(loan_data)) * 100
    loan_data = loan_data.loc[:, null_percent <= 50]

    # Add binary column for defaulted loans
    loan_data['default'] = loan_data['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default', 'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)'] else 0)

    # Add year column from issue date
    loan_data['issue_year'] = loan_data['issue_d'].str[-4:].astype(int)

    # Load unemployment rate data from multiple csv files and merge them on observation_date
    paths = unemployment_rate_data

    y_unemployment_df = pd.read_csv(paths[0])
    for path in paths[1:]:
        df = pd.read_csv(path)
        y_unemployment_df = y_unemployment_df.merge(df, on="observation_date", how="outer")

    # Add year column from observation date
    y_unemployment_df['year'] = y_unemployment_df['observation_date'].astype('datetime64[ns]').dt.year

    # Drop observation date column
    y_unemployment_df = y_unemployment_df.drop(columns=['observation_date'])

    # Import column names
    with open('data/unemployment_rate_dict.json') as f:
        states_ref = json.load(f)

    states_names = list(states_ref.keys())
    col_names = states_names + ["year"]

    # Rename columns
    y_unemployment_df.columns = col_names

    # Reshape unemployment data to long format
    y_unemployment_df = y_unemployment_df.melt(id_vars=["year"], var_name="state", value_name="y_unemployment_rate")

    # Merge unemployment rates with loan data
    data = loan_data.merge(
        y_unemployment_df,
        left_on=["issue_year", "addr_state"],
        right_on=["year", "state"],
        how="left"
    ).drop(columns=["year", "state"])

    # Drop unnecessary columns
    drop_cols = ['emp_title', 'issue_d', 'loan_status', 'url', 'title', 'zip_code', 'policy_code', 'earliest_cr_line', 'initial_list_status', 'last_pymnt_d', 'last_credit_pull_d']
    data = data.drop(columns=drop_cols)

    # Modify term column to keep only numeric part
    data['term'] = data['term'].apply(lambda x: x[:3])

    # Modify grade to numeric values
    grade_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    data['grade'] = data['grade'].map(grade_order)

    # Modify sub_grade to numeric values
    l_sub_grades = list(data['sub_grade'].value_counts().sort_index().index)
    # l_sub_grades.sort()
    d_sub_grades = {}
    for i, sub_grade in enumerate(l_sub_grades):
        d_sub_grades[sub_grade] = i + 1

    data['sub_grade'] = data['sub_grade'].map(d_sub_grades)

    # Modify emp_length to numeric values
    data['emp_length'] = (
        data['emp_length']
        .str.replace('< 1 year', '0')
        .str.replace('10+ years', '10')
        .str.replace('years', '')
        .str.replace('year', '')
        .astype(float)
    )

    # Convert pymnt_plan to numeric binary values
    data['pymnt_plan'] = data['pymnt_plan'].map({'n': 0, 'y': 1}).astype(int)

    # Convert hardship_flag to numeric binary values
    data['hardship_flag'] = data['hardship_flag'].map({'N': 0, 'Y': 1}).astype(int)

    # Convert debt_settlement_flag to numeric binary values
    data['debt_settlement_flag'] = data['debt_settlement_flag'].map({'N': 0, 'Y': 1}).astype(int)

    return data
