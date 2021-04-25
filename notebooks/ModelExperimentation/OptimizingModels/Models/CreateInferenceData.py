#takes csv input of fighter matchups and outputs csv with populated data

#Not a good approach to start with fighter_details, because there are a
#lot of modifications to details and attribute conversions

#what if I were to pass total_fight_data with FightNightMar20 addition and fighter_details to preprocess?
#FightNightMar20 entries would need to get additional data?

#


import numpy as np
import pandas as pd
import csv


def CreateInferenceData():
    fighter_df = pd.read_csv("../../../data/InferencingData/FightNightMar20.csv")





if __name__ == "__main__":
    CreateInferenceData()