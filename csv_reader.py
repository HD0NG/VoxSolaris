import pandas



"""
This file contains functions for loading csv file as dataframes. Including shadowmaps and pvdata + others if expanded
"""

def get_shadowdata(dense = False):


    """
    Returns dataframe with shadow data.

    3 current shadowmaps exist
    shadow_matrix.csv uses 5 degree step matrix
    shadow_matrix_1.csv uses 1 degree step matrix
    shadow_matrix_1106.csv has an improved matrix
    shadow_matrix_1606.csv has yet more improvements, this time blockage by the house itself is removed
    """

    if dense:
        # if dense, read 1 degree matrix
        df = pandas.read_csv("shadowdata/shadow_matrix_re.csv")
        df = df.drop(columns=["Unnamed: 0"])

    else:
        # if not dense, read 5 degree matrix
        df = pandas.read_csv("shadowdata/shadow_matrix.csv")
        df = df.drop(columns=["Unnamed: 0"])
        df.index = df.index*5


    return df



def get_year(year):

    year_txt = str(year-2000)

    df = pandas.read_excel("pvdata/pv_"+ year_txt +".xlsx", skiprows=[1], parse_dates=['Päivämäärä ja aika'])

    if df is None:
        return None

    """
     Index(['Päivämäärä ja aika', 'Energia | Symo 8.2-3-M (1)',
        'Energia MPP1 | Symo 8.2-3-M (1)', 'Energia MPP2 | Symo 8.2-3-M (1)',
        'Jännite AC L1 | Symo 8.2-3-M (1)', 'Jännite AC L2 | Symo 8.2-3-M (1)',
        'Jännite AC L3 | Symo 8.2-3-M (1)',
        'Jännite DC MPP1 | Symo 8.2-3-M (1)',
        'Jännite DC MPP2 | Symo 8.2-3-M (1)', 'Loisteho | Symo 8.2-3-M (1)',
        'Näennäisteho | Symo 8.2-3-M (1)', 'Ominaistuotto | Symo 8.2-3-M (1)',
        'Tehokerroin | Symo 8.2-3-M (1)', 'Virta AC, L1 | Symo 8.2-3-M (1)',
        'Virta AC, L2 | Symo 8.2-3-M (1)', 'Virta AC, L3 | Symo 8.2-3-M (1)',
        'Virta DC, MPP1 | Symo 8.2-3-M (1)',
        'Virta DC, MPP2 | Symo 8.2-3-M (1)', 'Aurinkosähkön tuotanto'],
       dtype='object')
     """

    #/ (5 / 60)

    df['Energia MPP1 | Symo 8.2-3-M (1)'] = df['Energia MPP1 | Symo 8.2-3-M (1)']/(5 / 60)
    # C (Energia MPP1 | Symo 8.2-3-M (1), 'Aurinkosähkön tuotanto'
    df.index = df["Päivämäärä ja aika"]
    df.index = pandas.to_datetime(df.index, format="%d.%m.%Y %H:%M", errors='coerce')

    # times are in local time for reason X
    # this bit of code should transform back into UTC time

    # Step 1: Localize to your local time zone
    df.index = df.index.tz_localize("Europe/Helsinki", ambiguous='NaT', nonexistent='NaT')

    # Step 2: Convert to UTC
    df.index = df.index.tz_convert("UTC")
    df.index = df.index.tz_convert(None)
    return df