import json
import pandas as pd

df_measure_dist = {
    "idx": [0, 2, 113, 1335, 1446, 2006, 2017, 2039, 2050, 2072, 2350, 2416, 2450, 2539, 2594],
    "Mouth_Opening": [20, 0, 0, 20, 20, 0, 0, 0, 0, 15, 0, 20, 15, 0, 0],
    "Left_Eye_Opening": [9, 4, 9, 9, 9, 9, 4, 9, 9, 9, 9, 5, 10, 5, 9],
    "Right_Eye_Opening": [9, 9, 9, 9, 4, 9, 4, 9, 9, 9, 9, 5, 10, 5, 9],
    "Smile_Width": [60, 50, 55, 55, 55, 40, 40, 50, 40, 60, 45, 60, 40, 45, 45]}

df = pd.DataFrame(df_measure_dist)

# Enregistrer dans un fichier CSV
df.to_csv("df_measure_dist.csv", index=False)
