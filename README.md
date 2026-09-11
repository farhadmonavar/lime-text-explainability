# Local Interpretable Model-Agnostic Explanations for Text Classification on the 20 Newsgroups Corpus

## Overview

This repository contains a self-contained Jupyter notebook (`lime_text.ipynb`) that investigates the application of **Local Interpretable Model-Agnostic Explanations (LIME)** to a binary text classification problem. The notebook is organized into two principal stages: (i) an automated model-selection procedure (AutoML) that trains and compares several ensemble and tree-based classifiers on TF–IDF representations of newsgroup documents, and (ii) a post-hoc interpretability stage in which LIME is used to generate local, human-readable explanations for individual predictions of the best-performing model.

The work is intended to serve as a pedagogical and experimental reference for the intersection of classical machine learning classifiers and model-agnostic interpretability methods, with a particular emphasis on the trade-off between predictive performance and explanatory transparency.

## Motivation

As machine learning models are increasingly deployed in decision-critical contexts, the ability to interpret and audit individual predictions has become a central concern in applied and theoretical machine learning research. LIME, introduced by Ribeiro, Singh, and Guestrin (2016), addresses this concern by approximating the decision boundary of an arbitrary "black-box" classifier in the local neighborhood of a specific instance using an interpretable surrogate model. This notebook applies that methodology to a text classification task, thereby providing a concrete, reproducible illustration of how local explanations can be extracted from complex, non-linear classifiers operating on high-dimensional sparse text features.

## Dataset

The experiments are conducted on a two-class subset of the **20 Newsgroups** dataset, a widely used benchmark corpus in text classification research. The two categories selected for this study are:

- `sci.electronics`
- `sci.space`

The dataset is expected to be present locally in the standard "bydate" train/test split format:

```
./Data/20news-bydate/20news-bydate-train
./Data/20news-bydate/20news-bydate-test
```

Documents are loaded via `sklearn.datasets.load_files` and are assumed to be encoded in Latin-1.

## Methodology

### 1. Feature Representation

Raw text documents are transformed into a numerical feature space using **Term Frequency–Inverse Document Frequency (TF–IDF)** vectorization (`sklearn.feature_extraction.text.TfidfVectorizer`), with case sensitivity preserved (`lowercase=False`).

### 2. Model Selection (AutoML Experiment)

To identify a suitable classifier for the subsequent interpretability analysis, five candidate models are trained on the TF–IDF representations and evaluated on the held-out test set using the binary F1-score:

| Classifier | Implementation |
|---|---|
| Random Forest | `sklearn.ensemble.RandomForestClassifier` |
| Bagging (k-NN base estimator) | `sklearn.ensemble.BaggingClassifier` with `KNeighborsClassifier` |
| Gradient Boosting | `sklearn.ensemble.GradientBoostingClassifier` |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` |
| Extra Trees | `sklearn.ensemble.ExtraTreesClassifier` |

Each model's performance is recorded, and the classifier achieving the highest F1-score is automatically selected as the champion model for downstream explanation. A manual override is also provided, allowing the user to disable the automated selection and specify a classifier of choice via a dropdown parameter.

### 3. Prediction Pipeline

The selected classifier is combined with the fitted TF–IDF vectorizer into a single inference pipeline (`sklearn.pipeline.make_pipeline`), enabling the model to operate directly on raw text input rather than pre-vectorized data. This step is a prerequisite for applying LIME, which perturbs and queries the model using raw text.

### 4. Local Explanation Generation

The `LimeTextExplainer` class from the `lime.lime_text` module is used to generate local explanations for individual test instances. For a selected document, the explainer:

1. Generates a set of perturbed variants of the input text by randomly removing words.
2. Queries the underlying classification pipeline for the predicted class probabilities of each perturbed variant.
3. Fits a locally weighted, interpretable linear model to approximate the classifier's behavior in the vicinity of the original instance.
4. Extracts the most influential words (features) contributing to the predicted class, along with their signed contribution weights.

Explanations are presented in three complementary formats:

- A textual list of feature–weight pairs (`exp.as_list()`)
- A horizontal bar-chart visualization (`exp.as_pyplot_figure()`)
- An interactive HTML rendering embedding highlighted text (`exp.as_html()`)

Additionally, the notebook includes a controlled ablation experiment, in which specific tokens are manually zeroed out of the feature vector to empirically verify their marginal contribution to the model's predicted probability, offering a sanity check against the explanations produced by LIME.

## Repository Structure

```
.
├── lime_text.ipynb      # Main notebook: training, model selection, and LIME analysis
├── Data/
│   └── 20news-bydate/
│       ├── 20news-bydate-train/
│       └── 20news-bydate-test/
└── README.md
```

## Requirements

The notebook depends on the following Python packages:

- `numpy`
- `scikit-learn`
- `lime`
- `matplotlib` (for `as_pyplot_figure`)
- `IPython` (for HTML rendering of explanations)

These may be installed via:

```bash
pip install numpy scikit-learn lime matplotlib ipython
```

## Usage

1. Ensure the 20 Newsgroups dataset is available locally under `./Data/20news-bydate/`, split into `20news-bydate-train` and `20news-bydate-test` subdirectories.
2. Launch the notebook environment:

   ```bash
   jupyter notebook lime_text.ipynb
   ```

3. Execute the cells sequentially. The AutoML stage will train and compare candidate classifiers; the subsequent cells will construct the prediction pipeline and generate LIME explanations for a user-specified document index.
4. Modify the `index` parameter in the "Selecting a text to explain" cell to inspect explanations for different documents, including two illustrative synthetic examples inserted into the test set for demonstration purposes.

## Interpretation of Results

The output of the interpretability stage should be read as a **local, additive, linear approximation** of the classifier's behavior around a single instance, and not as a global description of the model. Feature weights indicate the direction and relative magnitude of each word's contribution to the predicted class probability within the immediate neighborhood of the explained instance; they do not necessarily generalize to the model's behavior on the dataset as a whole.

## References

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.

Lang, K. (1995). NewsWeeder: Learning to Filter Netnews. *Proceedings of the Twelfth International Conference on Machine Learning*, 331–339.

## License

This repository is provided for research and educational purposes. Please consult the license file of this repository (if present) for terms of use, and note that the 20 Newsgroups dataset is subject to its own distribution terms.
