# Fashion Recommendation System Using Image Features (Hybrid CNN + GNN)

MSc Data Science dissertation, Kingston University London (2025-2026).

A content-based fashion recommender that suggests visually similar products from a query image. It extracts deep visual features with a pre-trained ResNet50, ranks items by cosine similarity, and tests whether a Graph Convolutional Network (GCN) built on a k-NN similarity graph can refine those embeddings with relational context. It uses no user ratings or purchase history, so it works for new products (the cold-start case).

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) by Param Aggarwal (Kaggle, 2018).

- 44,419 product images (JPG) with metadata from `styles.csv`
- Labels: gender, master category, sub-category, article type, base colour, season, year, usage

## Method

1. **Preprocessing:** cleaned the data and matched valid images to their metadata. Resized images to 224 x 224, converted them to RGB and normalised with ImageNet mean and standard deviation.
2. **Feature extraction:** a pre-trained ResNet50 (PyTorch) with the classification layer removed. The global average pooling output gives a 2,048-dimensional embedding per image, for all 44,419 images.
3. **CNN baseline:** nearest neighbours by cosine similarity on the ResNet50 embeddings.
4. **Graph construction:** a k-NN graph (k = 10, cosine similarity) over the first 10,000 items, where each product is a node and similar products are connected.
5. **GNN refinement:** a 2-layer GCN (PyTorch Geometric, 2048 -> 256 -> 128) produces 128-dimensional embeddings. It was trained for 10 epochs with Adam (learning rate 0.001) and a simple unsupervised MSE loss against the first 128 CNN feature dimensions.
6. **Evaluation:** Precision@K, Recall@K and NDCG@K on a 5,000-item subset. A recommendation counts as relevant when it has the same `masterCategory` as the query item.
7. **Visualisation:** t-SNE plots of the embeddings before and after GNN refinement.

## Results

| Model | Precision@5 | Recall@5 | NDCG@5 |
| --- | --- | --- | --- |
| CNN baseline (ResNet50 + cosine) | 0.983 | 0.996 | 0.992 |
| Hybrid CNN + GNN | 0.983 | 0.994 | 0.990 |

Both models perform almost identically on this task, and the GNN did not beat the CNN baseline on these metrics. ResNet50 features already separate broad product categories very well. Recall for the hybrid varies slightly between runs (0.992 to 0.994) because GNN training is stochastic.

In a qualitative comparison, the hybrid model's recommendations looked more category-consistent in some examples. This was judged by eye, not measured.

## Limitations

- Relevance is judged at master-category level (Apparel, Accessories, Footwear and so on), which is a coarse test. That is why both models score about 98%.
- The GNN was trained and evaluated on a 10,000-item subset (5,000 items for the metrics) because of Google Colab memory limits.
- The GCN was trained for only 10 epochs with a simple reconstruction-style loss and no hyperparameter search.
- Image features only: no product text, brand or user behaviour.

## Next steps

- Evaluate on finer labels (`articleType`, `baseColour`), where the baseline has more room to be wrong.
- Train on all 44,419 items.
- Try other GNN designs, such as Graph Attention Networks (GAT), and add category-based edges.
- Add text features (for example CLIP) and use approximate nearest-neighbour search (FAISS) for speed.

## Repository contents

| File | Description |
| --- | --- |
| `FinalDissertaton.ipynb` | Full notebook: preprocessing, feature extraction, graph construction, GCN training, evaluation |
| `image_metadata.csv` | Product metadata for all 44,419 images |
| `gnn_fashion_embeddings.npy` | GCN embeddings (10,000 x 128) |

`features_resnet50.npy` (about 355 MB, 44,419 x 2,048) is too large for GitHub. Download it from [Google Drive](ADD-YOUR-DRIVE-LINK-HERE), or regenerate it by running the feature-extraction section of the notebook.

## How to run

1. Open the notebook in Google Colab: [Open in Colab](https://colab.research.google.com/github/nidhishetty179-eng/fashion-recommender-cnn-gnn/blob/main/FinalDissertaton.ipynb)
2. Select a GPU runtime (Runtime > Change runtime type). The dissertation used an NVIDIA Tesla T4.
3. Upload your own Kaggle API key (`kaggle.json`) when prompted, to download the dataset. Never commit this key.
4. Run all cells.

## Tools

Python PyTorch Geometric, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, Google Colab (GPU)

## Author

Shreenidhi Shetty | [LinkedIn](https://www.linkedin.com/in/shreenidhi-shetty15)

Supervisor: Dr Xing Liang, Kingston University London.
