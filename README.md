# DistilBERT
## A Thematic Search Tool


### Description

This tool uses a lightweight variant of Google’s influential BERT Neural Network NLP model (DistilBERT) to classify our portfolio by undefined topics. The model has been trained on UKRI GTR data (2012-2022).


### Dependencies

The 'data' folder referenced in main.py is here:
https://ukri.sharepoint.com/:f:/r/sites/PolicyAnalysis/Shared%20Documents/Member%20Folders/Nicholas%20Hooper%20-%20Analyst/thematic%20search%20-%20tools/data?csf=1&web=1&e=8EgQ4e

The 'model' folder referenced in main.py is here:
https://ukri.sharepoint.com/:f:/r/sites/PolicyAnalysis/Shared%20Documents/Member%20Folders/Nicholas%20Hooper%20-%20Analyst/thematic%20search%20-%20tools/data?csf=1&web=1&e=8EgQ4e


### Basic Guide:

- Edit query variable with your input tokens that defines the topic area. Please use at least five terms (which can be more than one word), seperated by commas.
- Run all of main.py to create dataframe called 'df', which is a dataset containing all grants in the corpus, sorted from most relevant to least with a Cosine Similarity Score (CSS) ranging from 0-1.
- Any grants with a CSS below 0.5 should be removed as irrelevant.


### Contact

Nicholas Hooper - Strategic Analysis - UKRI Central
nicholas.hooper@ukri.org
