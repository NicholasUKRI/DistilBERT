# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
import base64
# import xlsxwriter
import io
import numpy as np
from search import *
import streamlit as st
from io import BytesIO
# from pyxlsb import open_workbook as open_xlsb
# from datetime import datetime
#import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime
import altair as alt
#from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json
import torch
from Google import Create_Service

# import pickle
# import os
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from tabulate import tabulate
#
# # If modifying these scopes, delete the file token.pickle.
# SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
#
# def get_gdrive_service():
#     creds = None
#     # The file token.pickle stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
#     # return Google Drive API service
#     return build('drive', 'v3', credentials=creds)
#
# def download():
#     service = get_gdrive_service()
#     # the name of the file you want to download from Google Drive
#     filename = 'metadata.json'
#     # search for the file by name
#     search_result = search(service, query=f"name='{filename}'")
#     # get the GDrive ID of the file
#     file_id = search_result[0][0]
#     # make it shareable
#     service.permissions().create(body={"role": "reader", "type": "anyone"}, fileId=file_id).execute()
#     # download file
#     download_file_from_google_drive(file_id, filename)
#
# download()


CLIENT_SECRET_FILE = 'credentials.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

@st.cache
def get_metadata():
    file_id = '113eOPDaBkcUv9jMMZjp1HRlsdGA-5Jmr'
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fd=fh, request = request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    metadataGoogle = fh.getvalue()
# turn bytes into JSON
    metadata = json.loads(metadataGoogle)
    return metadata









#
# #google key
# API_key = "AIzaSyDhEQ4z_j8AkBZJaS5elxyAMUzyU1ru2LE"
# #creating an instance of the class
# drive_service = build('drive', 'v2', developerKey = API_key)
#
# file_id = '113eOPDaBkcUv9jMMZjp1HRlsdGA-5Jmr'
# request = drive_service.files().get_media(fileId=file_id)
# fh = io.BytesIO()
# downloader = MediaIoBaseDownload(fh, request)
# done = False
# while done is False:
#         status, done = downloader.next_chunk()
#         print ("Download %d%%." % int(status.progress() * 100))
# metadataGoogle = fh.getvalue()
#
# # turn bytes into JSON
# metadata = json.loads(metadataGoogle)
#
# file_id2 = '1jDGcd3-gCBZyKxDz35hRJ4Z8CP3vPYWJ'
# request = drive_service.files().get_media(fileId=file_id2)
# gh = io.BytesIO()
#
# downloader = MediaIoBaseDownload(gh, request)
# done = False
# while done is False:
#     status, done = downloader.next_chunk()
#     print("Download %d%%." % int(status.progress() * 100))
# distilbert3ten = gh.getvalue()
#
#


st.set_page_config(page_title="UK Science R&D Search")

# converts dataframe to excel for export
def to_excel(df, query, min_words, min_threshold):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False,  startrow=4, sheet_name='Data')
    workbook = writer.book
    worksheet = writer.sheets['Data']
    format1 = workbook.add_format({'num_format': '0'})
    worksheet.set_column('A:A', None, format1)
    worksheet.write('A1', f"""Tokens:    {query}""")
    worksheet.write('A2', f"""Min Words:    {min_words} """)
    worksheet.write('A3', f"""Min distance:    {min_threshold}""")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Loads the vector embeddings
@st.cache
def load_embeddings():
    file_id = '1jDGcd3-gCBZyKxDz35hRJ4Z8CP3vPYWJ'
    requests = service.files().get_media(fileId=file_id)
    fg = io.BytesIO()
    downloader = MediaIoBaseDownload(fd=fg, request=requests)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    distilbert3ten = fg.getvalue()

    m = torch.load(io.BytesIO(distilbert3ten))
    return m


# Loads the tokenizer and masking model
@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Brawl/UKRI_DistilBERT")
    model = AutoModelForMaskedLM.from_pretrained("Brawl/UKRI_DistilBERT")
    return tokenizer, model


@st.cache
def load_indices():
    idx = torch.load("./data/chunkindices.pt")
    return idx

metadata = get_metadata()
embeddings = load_embeddings()
tokenizer, model = load_model()


def main():
    # with open("data/metadata.json", "r") as f:
    #     metadata = json.load(f)

    st.write('V1.0 April 20, 2022')
    st.title("What 🔬 science do we fund?")
    st.write('This search engine will sort the UKRI corpus from most relevant to least to your query...')

    # define query
    query = st.text_area("Topic (Use at least 5-10 words to describe your topic)", "")

    # embeddings = load_embeddings()
    # tokenizer, model = load_model()

    if query:
        # parameters
        col1, col2 = st.columns(2)
        with col1:
            min_words = st.slider("Min. words in abstract", 25, 250, value=75, step=25)
        with col2:
            min_threshold = st.slider("Min. similarity score", 0.5, 1.0, value=0.6, step=0.01)

        results = return_ranked(query, tokenizer, model, embeddings)

        # subset to enough words
        results = [r for r in results if len(metadata[str(r[0])]['abstract'].split()) > min_words]

        # return data
        meta = [{"ref": metadata[str(i)]["project_reference"],
                 "title": metadata[str(i)]["project_title"],
                 "value": int(metadata[str(i)]["value"]),
                 "n_words": len(metadata[str(i)]["abstract"].split()),
                 "start_date": int(datetime.strptime(metadata[str(i)]['start_date'],
                                                     "%d/%m/%Y").year),
                 "end_date": int(datetime.strptime(metadata[str(i)]['end_date'],
                                                   "%d/%m/%Y").year),
                 "distance": dist}
                for i, dist in results[:-1]]

        # sort for printing
        meta = sorted(meta, key=lambda x: -x["distance"])

        # Turn json file into dataframe
        df = pd.DataFrame(meta)
        df.insert(0, 'Rank', df.index + 1)  # Add new column (Rank) going from 1.

        df.rename(columns={'ref': 'Ref', 'title': 'Title', 'value': 'Value', 'n_words': 'Words', 'start_date': 'Start',
                           'end_date': 'End', 'distance': 'Distance'},
                  inplace=True)  # renaming columns

        # filter dataframe by min_threshold
        df = df[df.Distance >= min_threshold]

        df = df.explode('Ref')

        spark_data = df.groupby("Start").sum()
        spark_data = spark_data["Value"].to_list()
        year = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
        yearCheck = [2015, 2016, 2017, 2018, 2019, 2020, 2021] # for if statement later. Clean this up.
        yearCheck2 = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022] # for if statement later. Clean this up.

        def synthwave():
            #background = "#2e2157"  # dark blue-grey
            grid = "#2a3459"  # lighter blue-grey
            text = "#d3d3d3"  # light grey
            line_colors = [
                "#fe53bb",  # pink
            ]

            return {
                "config": {
                    "axis": {
                        "gridColor": grid,
                        "domainColor": None,
                        "tickColor": None,
                        "labelColor": text,
                        "titleColor": text
                    },
                    # "legend": {
                    #     "labelColor": text,
                    #     "titleColor": text
                    # },
                    "range": {
                        "category": line_colors
                    },
                    "area": {
                        "line": True,
                        "fillOpacity": 0.1
                    },
                    "line": {
                        "strokeWidth": 2
                    }
                }
            }
        alt.themes.register("synthwave", synthwave)
        alt.themes.enable("synthwave")



        # this if statement is disgusting, fix it with a function
        # check if 2015-2022 is in the data to create a chart, then remove 2022
        if set(yearCheck2).issubset(df['Start']):
            spark_data = spark_data[:-1]
            spark_data = np.round(np.divide(spark_data, 1000000), 2)
            data = {'Year': year, 'Value': spark_data}
            chart_data = pd.DataFrame(data)
            # Custom Altair line chart where you set color and specify dimensions
            custom_chart = alt.Chart(chart_data).mark_area().encode(
                x=alt.X('Year', axis=alt.Axis(title="")),
                y=alt.Y('Value', axis=alt.Axis(title="Value (£m)")),
                color=alt.value("#34D5AE")
            ).properties(
                width=700,
                height=200
            ).configure_axis(
                grid=False
            ).configure_view(
                strokeWidth=0
            )
            st.altair_chart(custom_chart)

        # check if 2015-2021 are in the data so I can at least create a chart!
        elif set(yearCheck).issubset(df['Start']):
            spark_data = np.round(np.divide(spark_data, 1000000), 2)
            data = {'Year': year, 'Value': spark_data}
            chart_data = pd.DataFrame(data)
            # Custom Altair line chart where you set color and specify dimensions
            custom_chart = alt.Chart(chart_data).mark_area().encode(
                x=alt.X('Year', axis=alt.Axis(title="")),
                y=alt.Y('Value', axis=alt.Axis(title="Value (£m)")),
                color=alt.value("#34D5AE")
            ).properties(
                width=700,
                height=200
            ).configure_axis(
                grid=False
            ).configure_view(
                strokeWidth=0
            )
            st.altair_chart(custom_chart)
        else:
            pass

        def format_currency(currency):
            currency_string = "£{:,.0f}".format(currency)
            return currency_string

        df["Value"] = df["Value"].apply(lambda x: format_currency(x))

        # Reference column is a list of grants. This will separate those grants out
        # df = df.explode('reference')

        #CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                    tbody th {display:none}
                    .blank {display:none}
                    </style>
                    """

        if st.button("Refine Algorithm"):
            st.text_area("Is this relevant?", value = df['Title'][0])

            col1, col2= st.columns([1,11])
            with col1:
                st.button("Yes")
            with col2:
                st.button("No")



        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        df_xlsx = to_excel(df, query, min_words, min_threshold)


        f"""
        Download all UKRI Grants with at least:
          * {min_words} words in the abstract
          * A similarity score of {min_threshold}
        """
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        st.download_button(label='📥 Download',
                           data=df_xlsx,
                           file_name=f'query_{dt_string}.xlsx')
        max_df = df['Rank'].max()
        df = df[df['Rank'] <= 100]  # filter to top 100 outputs
        max_rank = df['Rank'].max()  # get the max number of outputs after user change criteria

        st.write(f"""
            # Top Grants
            """)

        if math.isnan(max_rank):  # if max_rank is nan (i.e. there are no results), change to 0
            st.write("There are 0 results for your search")
        elif max_rank == 100:
            st.write(f"Showing top {max_rank} results, out of {max_df}")
        else:
            st.write(f"There are {max_rank} results for your search")

        st.table(df)


if __name__ == "__main__":
    main()


