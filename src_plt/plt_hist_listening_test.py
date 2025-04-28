import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the Excel file
file_path = "../data/listening_test_source/results_Tuni_BWE_N16_final_samplewise.xlsx"
df0 = pd.read_excel(file_path).dropna()
print(df0)
df = df0.iloc[1:7]
df_std_dev = df0.loc[24:29]

print(df0)
print(df_std_dev)
#input()

df.columns =["Audio", "Score"]
df["Audio"] = df["Audio"].astype(str).apply(lambda x: " ".join(x.split(" ")[1:]) if " " in x else [])

df_std_dev.columns =["Audio", "Score"]
df_std_dev["Audio"] = df_std_dev["Audio"].astype(str).apply(lambda x: " ".join(x.split(" ")[1:]) if " " in x else [])

df = df [df["Audio"].isin( ["Wideband", "Audiounet", "HifiGAN", "GAN", "LP 3.5kHz", "Spline-up 7kHz"])]
df_std_dev = df_std_dev [df_std_dev["Audio"].isin(["Wideband", "Audiounet", "HifiGAN", "GAN", "LP 3.5kHz", "Spline-up 7kHz"])]

print(df)
print(df_std_dev)

# Plot

#plt.figure(figsize=(10, 5))
#plt.bar(df["Audio"], df["Score"].astype(float), color="skyblue")

# Plot
plt.figure(figsize=(10, 5))
plt.bar(df["Audio"], df["Score"].astype(float), color="skyblue", yerr=df_std_dev["Score"].astype(float), capsize=5)

# Labels and title
#plt.xlabel("Audio Method", fontsize= 20)

plt.ylabel("Score", fontsize= 25)
#plt.title("Listening Test Scores", fontsize= 25)
plt.xticks(rotation=45, fontsize= 25)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0, fontsize= 25)  # Rotate x-axis labels for better readability

plt.tight_layout()
plt.savefig("../results/histograms48/listening_test_scores.png")

# Show the plot
plt.show()
