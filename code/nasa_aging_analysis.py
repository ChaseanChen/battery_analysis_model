import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_nasa_capacity(file_path):
    mat = scipy.io.loadmat(file_path)
    key = next(k for k in mat if not k.startswith("__"))
    cycles = mat[key][0, 0]["cycle"][0]

    caps = []
    for c in cycles:
        if c["type"][0] == "discharge":
            data = c["data"][0, 0]
            if "Capacity" in data.dtype.names:
                caps.append(float(data["Capacity"][0][0]))
    return np.asarray(caps)

if __name__ == "__main__":
    files = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]
    records = []

    plt.figure(figsize=(9, 5))
    for f in files:
        if not os.path.exists(f):
            continue

        caps = load_nasa_capacity(f)
        fade = (caps[0] - caps[-1]) / caps[0] * 100

        records.append({
            "Cell": f.replace(".mat", ""),
            "Cycles": len(caps),
            "Initial_Ah": round(caps[0], 3),
            "Final_Ah": round(caps[-1], 3),
            "Fade_%": round(fade, 2)
        })

        plt.plot(caps, lw=2, label=f.replace(".mat", ""))

    df = pd.DataFrame(records)
    print("\nNASA Aging Summary\n", df)

    plt.xlabel("Cycle Number")
    plt.ylabel("Capacity (Ah)")
    plt.title("NASA Li-ion Battery Capacity Fade")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Fig2.1_NASA_Capacity_Degradation.png", dpi=300)
    plt.show()
