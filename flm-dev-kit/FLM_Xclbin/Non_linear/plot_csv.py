# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("results.csv")

# Extract columns
x = data["x"]
o = data["o"]
o_ref = data["o_ref"]

plt.figure(figsize=(10, 6))
plt.plot(x, o_ref, label="Reference Output", linewidth=2)
plt.plot(x, o, 'o', markersize=2, label="Kernel Output")

plt.title("Comparison of Reference and Kernel Outputs", fontsize=16)
plt.xlabel("Input (x)", fontsize=14)
plt.ylabel("Output", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("results_plot.png", dpi=300)
plt.show()