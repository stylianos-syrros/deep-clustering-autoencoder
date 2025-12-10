# Deep Clustering with Autoencoders

This project explores how deep feature learning and dimensionality reduction affect unsupervised clustering performance.

We apply **K-Means** and **Agglomerative Hierarchical Clustering** both on:
- the original feature space, and
- a latent space learned by an Autoencoder

Clustering quality is evaluated using **Purity** and **F-measure**.

---

## 🚀 Project Highlights

- **Fully-connected Autoencoder** with bottleneck sizes \( M \in \{2, 10, 50\} \)
- **Comparison** of clustering performance before and after dimensionality reduction
- **Custom implementation** of Purity and F-measure
- **Visualization** of the 2-D latent embedding
- **Complete analysis** included in the PDF report

---

## 📁 Files

- **`set2_4805.py`**  
  Main experiment script. It:
  - Loads `train.csv`
  - Trains an Autoencoder
  - Generates latent representations
  - Runs K-Means & Agglomerative Clustering
  - Computes Purity & F-measure
  - Visualizes the 2-D latent space when `M = 2`

- **`train.csv`**  
  Dataset used for training and evaluation:
  - 20 input features
  - 1 label (`price_range`) used only for scoring

- **`Deep Clustering with Autoencoders.pdf`**  
  Full technical report containing:
  - Dataset description
  - Clustering methods
  - Autoencoder architecture
  - Experimental setup
  - Results & discussion

---

## ⚙️ Requirements

Developed using **Python 3.10** with the following main dependencies:

- numpy
- pandas
- scikit-learn
- matplotlib
- tensorflow==2.10.0
- protobuf<=3.20.3

Install via:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Place `train.csv` next to `set2_4805.py`.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the experiment:

   ```bash
   python set2_4805.py
   ```

The script prints:
- Purity
- F-measure
- Results for different values of `K` and `M`

and displays a scatter plot for the **`M = 2`** latent space.

---

## 📌 Workflow Overview

```text
       ┌───────────────┐
       │   train.csv    │
       └───────┬────────┘
               │
     ┌─────────▼──────────┐
     │ Original Clustering │  (K-Means, Agglomerative)
     └─────────┬──────────┘
               │
   ┌───────────▼────────────┐
   │     Autoencoder (AE)    │
   └───────────┬────────────┘
               │
    ┌──────────▼───────────┐
    │   Latent Embeddings   │  (M = 2, 10, 50)
    └──────────┬───────────┘
               │
   ┌───────────▼────────────┐
   │ Clustering in AE Space  │
   └─────────────────────────┘
```