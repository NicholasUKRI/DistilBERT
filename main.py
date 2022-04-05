from search import *

# Any grants with abstracts less than min_words will be removed. Default = 75
min_words = 75

# Query are tokens which represent the topic area. This is a string with tokens seperated
# by commas. Minimum tokens = 5
query = "solar cell, offshore, energy, energy sector, energy generation, shale gas, nuclear fission, " \
        "fuel cell, fossil fuel, bioenergy, renewable energy, geothermal, nuclear fusion, solar power, wind power, " \
        "photovoltaic, energy storage, energy efficiency"

df = run_tool(query, min_words)
