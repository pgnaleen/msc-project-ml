from utils import *


def prep_data(file_name):
    # Use a breakpoint in the code line below to debug your script.
    # Load the dataset into a dataframe
    # df = pd.read_csv("CIDDS-001-external-week1.csv")
    df = pd.read_csv(file_name)

    # Print the top 5 rows
    print(df.head())

    # Class is our target column and our target column have 3 unique values, as printed.
    df['class'].value_counts()

    # Drop these columns as data is missing in these columns.
    df = df.drop(['attackType', 'attackID', 'attackDescription'], axis=1)

    # print the rest of the columns
    print(df.columns)

    # Print if our dataset have any null or empty rows.
    print(df.isnull().sum())

    # List of the columns we want to update from text to number
    columns = ['class', 'Proto', 'Src IP Addr', 'Dst IP Addr', 'Bytes', 'Flags']

    # Calling function for each column, one by one
    for col in columns:
        # Replace_data function takes two argument as described above.
        # 1 -  unique values on a column.
        # 2 -  Numbers list, so function can replace text with numbers
        # 3 -  Col is the column name, comes from the columns list
        replace_data(list(df[col].unique()), list(range(0, len(list(df[col].unique())))), col, df)

    # print the top 5 rows to see the changes
    print(df.head())

    # We want to get the year, month, day, hour, minute as second as separate columns, so we will make date as datetime
    # object Because
    # to make a pattern of date transformation, like by each row there is a change in time, so model learn from this
    # division of timeframe if we keep it as single colum, we have to make it a date index and it doesn't help in
    # learning includes date time as well with this recognition
    # it gives trends to data flow
    df['Date first seen'] = pd.to_datetime(df['Date first seen'])

    df['Date first seen'] = pd.to_datetime(df['Date first seen'], format='%Y%m%d')

    # Here we are extratcing year,month and day as new columns, in each rows
    df['year'] = pd.DatetimeIndex(df['Date first seen']).year
    df['month'] = pd.DatetimeIndex(df['Date first seen']).month
    df['day'] = pd.DatetimeIndex(df['Date first seen']).day
    df['hour'] = pd.DatetimeIndex(df['Date first seen']).hour
    df['min'] = pd.DatetimeIndex(df['Date first seen']).minute
    df['sec'] = pd.DatetimeIndex(df['Date first seen']).second

    # Now we extract all columns as new, now we can drop the actual datetime column
    # here inplace=True because......
    # inplace true means update the dataframe with these changes
    df.drop(['Date first seen'], axis=1, inplace=True)

    # print the top 5 rows to see the changes of datetime
    print(df.head())

    # Save this process data as csv file, to be used for the models
    df.to_csv("Processed_data.csv")
