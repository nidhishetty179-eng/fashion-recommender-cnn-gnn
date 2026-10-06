# Fashion Recommendation System Using Image Features (Hybrid CNN + GNN)

MSc Data Science dissertation, Kingston University London (2025-2026).

An image-based recommender that suggests visually similar fashion products. It extracts visual features with a pre-trained ResNet50, ranks items by cosine similarity, and tests whether a Graph Neural Network (GNN) built on a k-NN similarity graph improves the results.

## Dataset

Fashion Product Images (Small) from Kaggle: 44,419 product images with metadata (gender, category, article type, colour, season, year, usage).

## Method

1. ResNet50 (PyTorch) extracts a 2,048-dimensional feature vector from each image.
2. Baseline: nearest neighbours using cosine similarity on the CNN features.
3. Hybrid: a k-NN similarity graph over 10,000 items, with a GNN (PyTorch Geometric) that learns 128-dimensional embeddings.
4. Evaluation: Precision@K, Recall@K and NDCG@K on a 5,000-item subset. An item counts as relevant when it has the same master category as the query item.

## Results

| Model | Precision@5 |
| --- | --- |
| CNN baseline | 0.983 |
| Hybrid CNN + GNN | 0.983 |

The hybrid model reached a Recall@5 of 0.993.

## Limitations

- Relevance is judged by broad category (Apparel, Accessories, Footwear), which is a coarse test, so both models score about 98%.
- The GNN was trained on a 10,000-item subset because of Colab memory limits.
- Next step: test on finer labels such as article type and colour.

## Tools

Python, PyTorch, PyTorch Geometric, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, Google Colab

## Author

Shreenidhi Shetty - [LinkedIn](https://www.linkedin.com/in/shreenidhi-shetty15)
