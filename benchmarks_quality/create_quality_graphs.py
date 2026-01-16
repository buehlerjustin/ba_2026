#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.DataFrame(
        {
            "Komparator": ["dl", "jw", "ngram"],
            "Recall": [0.580, 0.916, 0.351],
            "F1": [0.734, 0.956, 0.520],
        }
    ).set_index("Komparator")

    ax = df[["Recall", "F1"]].plot(kind="bar")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wert")
    ax.set_title("Recall und F1 pro Komparator (10k)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("./benchmarks_quality/recall_f1_10k.pdf")
    plt.savefig("./benchmarks_quality/recall_f1_10k.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
