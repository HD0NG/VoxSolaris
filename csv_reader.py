import pandas

# import pandas as pd
# import numpy as np

# from pandas.errors import AmbiguousTimeError
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

# def get_year(year: int) -> pd.DataFrame | None:
#     year_txt = str(year - 2000)
#     df = pd.read_excel(
#         f"pvdata/pv_{year_txt}.xlsx",
#         skiprows=[1],
#         parse_dates=["Päivämäärä ja aika"],
#         # If decimals use commas in your export, uncomment:
#         # decimal=",",
#     )
#     if df is None or df.empty:
#         return None

#     ts_col = "Päivämäärä ja aika"
#     e_mpp1_col = "Energia MPP1 | Symo 8.2-3-M (1)"

#     # Ensure datetime index (Finnish format often parsed already by read_excel)
#     df = df.set_index(ts_col)
#     df.index = pd.to_datetime(df.index, errors="coerce")



#     # Local time → UTC (handles DST). Prefer infer for ambiguous transitions.
#     # df.index = df.index.tz_localize("Europe/Helsinki", ambiguous="infer", nonexistent="shift_forward")
#     try:
#     # try the simple, DST-aware path first
#         df.index = df.index.tz_localize("Europe/Helsinki",
#                                         ambiguous="infer",
#                                         nonexistent="shift_forward")
#     except Exception:
#         # build an explicit ambiguous mask:
#         # for duplicated wall-times (only occur at fall-back), mark the FIRST as DST (True),
#         # the SECOND as standard time (False)
#         is_second_occurrence = df.index.duplicated(keep="first")
#         ambiguous_mask = ~is_second_occurrence

#         df.index = df.index.tz_localize("Europe/Helsinki",
#                                         ambiguous=ambiguous_mask,
#                                         nonexistent="shift_forward")

#     # convert to naive UTC if you want to keep the index timezone-free
#     df.index = df.index.tz_convert("UTC").tz_convert(None)

#     # df.index = df.index.tz_convert("UTC").tz_convert(None)  # naive UTC

#     # Make sure energy column is numeric
#     df[e_mpp1_col] = pd.to_numeric(df[e_mpp1_col], errors="coerce")

#     # If it's per-interval Wh, convert to average W using actual interval length
#     dt_hours = df.index.to_series().diff().dt.total_seconds() / 3600.0
#     # If the first delta is NaN, forward-fill with the median interval length
#     dt_hours.iloc[0] = dt_hours.median(skipna=True)

#     df["P_MPP1_W"] = (df[e_mpp1_col] / dt_hours).clip(lower=0)  # Wh / h = W

#     # Optional: also compute kW and keep the original Wh
#     df["P_MPP1_kW"] = df["P_MPP1_W"] / 1000.0

#     # Handy aggregations
#     df["E_MPP1_kWh"] = df[e_mpp1_col] / 1000.0
#     hourly_energy_kWh = df["E_MPP1_kWh"].resample("H").sum(min_count=1)
#     daily_energy_kWh  = df["E_MPP1_kWh"].resample("D").sum(min_count=1)

#     # Return the full df; you can also return the aggregates if you like
#     return df