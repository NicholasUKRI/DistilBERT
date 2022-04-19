import base64
import xlsxwriter
from search import *
import streamlit as st
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import math

st.set_page_config(page_title="UK Science R&D Search")


# converts dataframe to excel for export
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# Loads the vector embeddings
@st.cache
def load_embeddings(embeddings_path="./data/distilbert3tensor.pt"):
    m = torch.load(embeddings_path)
    return m


# Loads the tokenizer and masking model
@st.cache(allow_output_mutation=True)
def load_model(path_or_name='./model/distilbert3/'):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForMaskedLM.from_pretrained(path_or_name)
    return tokenizer, model


@st.cache
def load_indices():
    idx = torch.load("./data/chunkindices.pt")
    return idx



# #
# def write_paper_table(data, n_words=True, distance=True):
#     """
#   Writes a markdown table of papers and their titles.
#   :param data:
#   :param n_words: bool; if True, add column with abstract word count.
#   """
#     table = f"""
#   |Rank|Title|Value|{"# words|" * n_words}{"Similarity|" * distance}
#   |--|--|--|--|--|
#   """
#
#     for i, el in enumerate(data):
#         line = f"""|{i + 1}|**{el["title"]}**|£{el["value"]}|"""
#         if n_words:
#             line += f"""{str(el["n_words"])}|"""
#         if distance:
#             line += f"""{str(round(el["distance"], 2))}"""
#         line = f"""{line}
#             """
#         table += line
#
#     st.markdown(table)


def main():
    with open("data/metadata.json", "r") as f:
        metadata = json.load(f)

    st.title("What 🔬 science do we fund?")
    st.write('This search engine will sort the UKRI corpus from most relevant to least to your query...')

    # # Any grants with abstracts less than min_words will be removed. Default = 75
    # min_words = 75

    # define query
    query = st.text_area("Topic (Use at least 5-10 words to describe your topic)", "")

    embeddings = load_embeddings()
    tokenizer, model = load_model()

    if query:
        # parameters
        col1, col2 = st.columns(2)
        with col1:
            min_words = st.slider("Min. words in abstract", 25, 250, value=75, step=25)
        with col2:
            min_threshold = st.slider("Min. similarity score", 0.5, 1.0, value=0.5, step=0.01)

        results = return_ranked(query, tokenizer, model, embeddings)

        # subset to enough words
        results = [r for r in results if len(metadata[str(r[0])]['abstract'].split()) > min_words]

        # return data
        meta = [{"ref": metadata[str(i)]["project_reference"],
                 "title": metadata[str(i)]["project_title"],
                 "value": int(metadata[str(i)]["value"]),
                 "n_words": len(metadata[str(i)]["abstract"].split()),
                 # "date": int(datetime.strptime(metadata[str(i)]['start_date'],
                 #                               "%d/%m/%Y").year),
                 "distance": dist}
                for i, dist in results[:-1]]

        # sort for printing
        meta = sorted(meta, key=lambda x: -x["distance"])

        # Turn json file into dataframe
        df = pd.DataFrame(meta)
        df.insert(0, 'Rank', df.index + 1)  # Add new column (Rank) going from 1.

        df.rename(columns={'ref': 'Ref', 'title': 'Title', 'value': 'Value', 'n_words': 'Words', 'distance': 'Distance'},
                  inplace=True)  # renaming columns

        # filter dataframe by min_threshold
        df = df[df.Distance >= min_threshold]

        def format_currency(currency):
            currency_string = "£{:,.0f}".format(currency)
            return currency_string

        df["Value"] = df["Value"].apply(lambda x: format_currency(x))

        # Reference column is a list of grants. This will separate those grants out
        # df = df.explode('reference')

        # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                    tbody th {display:none}
                    .blank {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        df = df.explode('Ref')

        df_xlsx = to_excel(df)


        f"""
        Download all UKRI Grants with at least:
          * {min_words} words in the abstract
          * A similarity score of {min_threshold}
        """

        st.download_button(label='📥 Download',
                           data=df_xlsx,
                           file_name='df_test.xlsx')

        df = df[df['Rank'] <= 100] # filter to top 100 outputs
        max_rank = df['Rank'].max() # get the max number of outputs after user change criteria

        st.write(f"""
            # Top Grants
            """)

        if math.isnan(max_rank): # if max_rank is nan (i.e. there are no results), change to 0
            st.write("There are 0 results for your search")
        elif max_rank == 100:
            st.write(f"Showing top {max_rank} results")
        else:
            st.write(f"There are {max_rank} results for your search")

        st.table(df)


if __name__ == "__main__":
    main()

#
# print(format_currency(100000, currency="GBP", locale="en_GB"))
# currency_string = "${:,.0f}".format(100000)
# currency_string
# from babel.numbers import format_currency
#
#         df["Value"] = df["Value"].apply(lambda x: format_currency(x, currency="GBP", locale="en_GB")
#
#
#                                                                   currency_digits=False=False))