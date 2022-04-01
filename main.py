import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
from search import *

## Functions
def load_embeddings(embeddings_path="data\\distilbert3tensor.pt"):
    M = torch.load(embeddings_path)
    return M

def load_model(path_or_name="model\\distilbert3"):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForMaskedLM.from_pretrained(path_or_name)
    return tokenizer, model

# Open metadata.JSON file containing grants and load
f = open("data\\metadata.json")
metadata = json.load(f)

# select number of relevant papers
num_results = 100000 # set to max currently
min_words = 75 # anything under 75 words is removed.

# define query, use at least 5-10 words to describe the topic
query = "solar cell, offshore, energy, energy sector, energy generation, shale gas, nuclear fission, fuel cell, fossil fuel, bioenergy, renewable energy, geothermal, nuclear fusion, solar power, wind power, photovoltaic, energy storage, energy efficiency"

embeddings = load_embeddings()
tokenizer, model = load_model()

# fetch results
results = return_ranked(query, tokenizer, model, embeddings)

# subset to enough words
results = [r for r in results if len(metadata[str(r[0])]['abstract'].split()) > min_words]

# return data
meta = [{"reference": metadata[str(i)]["project_reference"],
         "title": metadata[str(i)]["project_title"],
         "abstract": metadata[str(i)]["abstract"],
         "value": int(metadata[str(i)]["value"]),
         "n_words": len(metadata[str(i)]["abstract"].split()),
         "start_year": int(datetime.strptime(metadata[str(i)]['start_date'],
                                              "%d/%m/%Y").year),
         "end_year": int(datetime.strptime(metadata[str(i)]['end_date'],
                                             "%d/%m/%Y").year),
         "distance": dist}
        for i, dist in results[:num_results]]

# turn json file into dataframe
df = pd.DataFrame(meta)

#reference column is a list of grants as some grants have the same same titles, so explode the references down
df = df.explode('reference')
