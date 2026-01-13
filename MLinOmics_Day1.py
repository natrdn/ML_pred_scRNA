#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## ML in Omics - Practical Session 1: Complete Exercise Solutions
# Date: January 12, 2026  
# Topic: ML basics with sklearn - Cell Type Prediction  
# GitHub Repository: https://github.com/csbg/ML_in_omics

## Exercise Objectives
# This practical session aims to:
# 1. Set up your Python environment for ML in omics
# 2. Load and explore single-cell RNA-seq data using AnnData
# 3. Preprocess omics data (normalization and filtering)
# 4. Train a logistic regression model using scikit-learn
# 5. Evaluate model performance with proper train/test splitting
# 6. Understand the complete ML workflow for cell type prediction


# In[1]:


### Setup and Loading Data
# Import packages 
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Sanity check
import sys
print(sys.executable)


# In[3]:


# Import data
data = anndata.read_h5ad("data/my_dataset_small.h5ad")


# In[4]:


# Data overview
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(data)

print(f"\nNumber of cells: {data.n_obs}")
print(f"Number of genes: {data.n_vars}")
print(f"\nData matrix type: {type(data.X)}")
print(f"Data matrix shape: {data.X.shape}")


# In[5]:


# Priting data matrix
print("\n" + "=" * 50)
print("DATA MATRIX (X)")
print("=" * 50)

print(f"Shape: {data.X.shape}")
print(f"Type: {type(data.X)}")
print(f"Data type: {data.X.dtype}")

print("\nFirst 5 cells, first 5 genes:")
if hasattr(data.X, "toarray"):
    print(data.X[:5, :5].toarray())
else:
    print(data.X[:5, :5])


# In[6]:


# Priting cell metadata
print("\n" + "=" * 50)
print("CELL METADATA (obs)")
print("=" * 50)

print(f"Shape: {data.obs.shape}")
print("\nAvailable columns:")
print(list(data.obs.columns))

print("\nFirst 5 rows:")
data.obs.head()


# In[7]:


# Printing gene metadata
print("\n" + "=" * 50)
print("GENE METADATA (var)")
print("=" * 50)

print(f"Shape: {data.var.shape}")
print("\nAvailable columns:")
print(list(data.var.columns))

data.var.head()


# In[8]:


# Assessing cell type collumn
possible_columns = [
    "cell_type", "celltype", "cell_ontology_class",
    "cluster", "louvain", "leiden", "annotation"
]

celltype_column = None
for col in possible_columns:
    if col in data.obs.columns:
        celltype_column = col
        break

if celltype_column:
    print(f"\n✓ Cell type column found: '{celltype_column}'")
    print(f"Unique cell types: {data.obs[celltype_column].nunique()}")
    print("\nCell type distribution:")
    print(data.obs[celltype_column].value_counts())
else:
    print("\n⚠ Warning: Cell type column not found automatically.")
    print("Available columns:", list(data.obs.columns))


# In[9]:


# Plotting cell type distribution
data.obs[celltype_column].value_counts().plot.bar(figsize=(6,4))
plt.ylabel("Number of cells")
plt.title("Cell type distribution")
plt.tight_layout()
plt.show()


# In[10]:


### Data Processing
# Step-by-step manual normalization and filtering

print("=== Step-by-step normalization (per guide) ===")

# Save raw counts BEFORE normalization (for overview table/plot later)
data.layers["counts"] = data.X.copy()

# Normalization
# 1) Divide by sum of reads per cell
try:
    data.X = data.X / data.X.sum(axis=1)
except Exception:
    # Guide workaround for CSR/CSC issues
    data.X = (data.X / data.X.sum(axis=1)).tocsr()

# 2) Multiply by 1 million (1e6)
data.X = data.X * 1e6

# 3) Log normalize
data.X = np.log1p(data.X)

# 4) Convert to CSR
if hasattr(data.X, "tocsr"):
    data.X = data.X.tocsr()

# Visuals normalization (table + plot)

print("\n=== Overview AFTER normalization ===")

raw_vals = data.layers["counts"].toarray().ravel() if hasattr(data.layers["counts"], "toarray") else data.layers["counts"].ravel()
norm_vals = data.X.toarray().ravel() if hasattr(data.X, "toarray") else data.X.ravel()

summary_norm = pd.DataFrame({
    "stage": ["raw_counts", "normalized"],
    "min":   [raw_vals.min(),  norm_vals.min()],
    "max":   [raw_vals.max(),  norm_vals.max()],
    "mean":  [raw_vals.mean(), norm_vals.mean()],
    "median":[np.median(raw_vals), np.median(norm_vals)],
    "std":   [raw_vals.std(),  norm_vals.std()],
})
display(summary_norm)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.hist(raw_vals, bins=50)
plt.title("Raw counts")
plt.xlabel("Expression")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(norm_vals, bins=50)
plt.title("After normalization")
plt.xlabel("Expression")
plt.tight_layout()
plt.show()

# Filtering

print("\n=== Gene filtering (>=10% of cells) ===")

n_cells = data.shape[0]
gene_sums = data.X.sum(axis=0)

# flatten gene_sums to 1D
# gene_sums = gene_sums.A1 if hasattr(gene_sums, "A1") else np.asarray(gene_sums).ravel()

keep = gene_sums > (0.10 * n_cells)
data = data[:, keep].copy()

print(f"Genes kept: {keep.sum()} / {len(keep)}")
print(f"New shape: {data.shape}")

# Visuals filtering (table + plot)

print("\n=== Overview AFTER filtering ===")

filt_vals = data.X.toarray().ravel() if hasattr(data.X, "toarray") else data.X.ravel()

summary_filt = pd.DataFrame({
    "stage": ["raw_counts", "normalized", "filtered"],
    "min":   [raw_vals.min(),   norm_vals.min(),   filt_vals.min()],
    "max":   [raw_vals.max(),   norm_vals.max(),   filt_vals.max()],
    "mean":  [raw_vals.mean(),  norm_vals.mean(),  filt_vals.mean()],
    "median":[np.median(raw_vals), np.median(norm_vals), np.median(filt_vals)],
    "std":   [raw_vals.std(),   norm_vals.std(),   filt_vals.std()],
})
display(summary_filt)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.hist(raw_vals, bins=50)
plt.title("Raw counts")
plt.xlabel("Expression")
plt.ylabel("Frequency")

plt.subplot(1,3,2)
plt.hist(norm_vals, bins=50)
plt.title("After normalization")
plt.xlabel("Expression")

plt.subplot(1,3,3)
plt.hist(filt_vals, bins=50)
plt.title("After filtering")
plt.xlabel("Expression")

plt.tight_layout()
plt.show()


# In[11]:


# Training ML Model
# Extract Features (X) and Labels (Y)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Determine cell type column (update this if different)
celltype_column = 'cell_type'  # Adjust based on your dataset

# Extract features (gene expression matrix)
X = data.X
if hasattr(X, 'toarray'):
    X = X.toarray()  # Convert sparse to dense

print(f"Feature matrix (X) shape: {X.shape}")
print(f"Type: {type(X)}")

# Extract labels (cell types)
Y_raw = data.obs[celltype_column].values
print(f"\nLabel array (Y) shape: {Y_raw.shape}")
print(f"Unique cell types: {np.unique(Y_raw)}")
print(f"Number of cell types: {len(np.unique(Y_raw))}")

# Encode labels as integers (sklearn requires this for some algorithms)
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y_raw)

print(f"\nLabel encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} -> {i}")


# In[12]:


# Subset the data in Train and Test
# Split data: 70% train, 30% test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42,
    stratify=Y  # Maintain class proportions
)

print("\n" + "=" * 50)
print("TRAIN/TEST SPLIT")
print("=" * 50)
print(f"Total samples: {X.shape[0]}")
print(f"Training samples: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Test samples: {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

# Check class distribution
print(f"\nClass distribution in training set:")
for i, label in enumerate(label_encoder.classes_):
    count = (Y_train == i).sum()
    print(f"  {label}: {count} ({count/len(Y_train)*100:.1f}%)")

print(f"\nClass distribution in test set:")
for i, label in enumerate(label_encoder.classes_):
    count = (Y_test == i).sum()
    print(f"  {label}: {count} ({count/len(Y_test)*100:.1f}%)")


# In[13]:


# Training Logistic Regression 
from sklearn.linear_model import LogisticRegression

print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

# Design the model
model = LogisticRegression(
    max_iter=1000,      # Maximum iterations for convergence
    random_state=42,    # For reproducibility
    verbose=1,          # Show training progress
    n_jobs=-1           # Use all CPU cores
)

print("Training logistic regression model...")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(label_encoder.classes_)}")

# Train the model
model.fit(X_train, Y_train)

print("\n✓ Model training complete!")
print(f"Converged: {model.n_iter_}")
print(f"Model coefficients shape: {model.coef_.shape}")


# In[14]:


# Checking the model parameters
# Examine learned weights
print("\n" + "=" * 50)
print("MODEL PARAMETERS")
print("=" * 50)

print(f"Number of classes: {len(model.classes_)}")
print(f"Classes: {model.classes_}")
print(f"\nCoefficient matrix shape: {model.coef_.shape}")
print(f"  (rows = classes, columns = genes)")
print(f"\nIntercept shape: {model.intercept_.shape}")

# Find most important genes for each class (top weights)
for i, class_label in enumerate(label_encoder.classes_):
    weights = model.coef_[i]
    top_gene_indices = np.argsort(np.abs(weights))[-5:][::-1]
    
    print(f"\nTop 5 genes for predicting '{class_label}':")
    for rank, gene_idx in enumerate(top_gene_indices, 1):
        gene_name = data.var_names[gene_idx]
        weight = weights[gene_idx]
        print(f"  {rank}. {gene_name}: weight = {weight:.4f}")


# In[15]:


# Assessing perfomance on the test subset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

# Predict on training subset (Where we fit the model: So seen data)
Y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, Y_train_pred)

# Predict on test subset (Unseen data)
Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print(f"Seen - > Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Unseen -> Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check for overfitting
if train_accuracy - test_accuracy > 0.1:
    print("\n⚠ Warning: Large gap between train and test accuracy suggests overfitting!")
else:
    print("\n✓ Model generalizes well to unseen data")


# In[16]:


# Detailed test performance 

import sklearn.metrics as skm
from math import ceil
from scipy import stats

# Predictions on test set
Y_test_hat = model.predict(X_test)

# Decode integer labels back to cell type strings
Y_test_lbl = label_encoder.inverse_transform(Y_test)
Y_test_hat_lbl = label_encoder.inverse_transform(Y_test_hat)

print("Overall test accuracy:", skm.accuracy_score(Y_test_lbl, Y_test_hat_lbl))

# Per-class accuracy table (exactly the structure your professor describes)
eval = []
for ct in set(Y_test_lbl):
    idx = [i for i, cx in enumerate(Y_test_lbl) if cx == ct]
    res = pd.DataFrame({
        "cell_type": ct,
        "accuracy": sum([Y_test_hat_lbl[i] == ct for i in idx]) / len(idx),
        "number": len(idx)
    }, index=[0])
    eval.append(res)

eval = pd.concat(eval, ignore_index=True)
print(eval.sort_values(["accuracy", "number"], ascending=[True, False]))


# In[17]:


# Visuals -> Plot: accuracy (x) vs cell_type (y)
plt.figure(figsize=(6, max(4, 0.25 * eval.shape[0])))
plt.scatter(eval.accuracy, eval.cell_type)
plt.xlabel("Accuracy (per class)")
plt.ylabel("Cell type")
plt.title("Per-class accuracy (easy task: cell_type)")
plt.show()

# Visuals -> Plot: accuracy vs number of cells + Pearson correlation
plt.figure(figsize=(6, 4))
plt.scatter(eval.number, eval.accuracy)
plt.xlabel("Number of test cells in class")
plt.ylabel("Accuracy (per class)")
plt.title("Accuracy vs number of cells (easy task)")
plt.show()

r, p = stats.pearsonr(eval.number, eval.accuracy)
print(f"Pearson r = {r:.3f}, p = {p:.3g}")


# In[18]:


# Classification report
print("\nClassification report (easy task):")
print(skm.classification_report(Y_test_lbl, Y_test_hat_lbl, digits=3))


# In[19]:


# Visuals -> ROC curves + AUC for each cell type (one-vs-rest)
Y_test_hat_probs = model.predict_proba(X_test)
classes_enc = model.classes_  # encoded labels (ints)
classes_lbl = label_encoder.inverse_transform(classes_enc)

n_classes = len(classes_lbl)
n_cols = 3
n_rows = ceil(n_classes / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True, figsize=(4*n_cols, 3*n_rows))
axs = np.array(axs).flatten()

print("\nAUC per cell type (easy task):")
for i, cl_name in enumerate(classes_lbl):
    fpr, tpr, thresholds = skm.roc_curve(
        [0 + (x == cl_name) for x in Y_test_lbl],
        Y_test_hat_probs[:, i]
    )
    axs[i].plot(fpr, tpr)
    axs[i].set_title(cl_name, fontdict={'color': 'blue', 'size': 7})
    auc_val = skm.auc(fpr, tpr)
    print(cl_name, auc_val)

# turn off unused axes
for j in range(n_classes, len(axs)):
    axs[j].axis("off")

plt.show()


# In[20]:


# Save the predictions "Easy Model"
row_id = np.arange(len(Y_test))

Y_test_hat = model.predict(X_test)

easy_pred_df = pd.DataFrame({
    "row_id": row_id,
    "true_cell_type": label_encoder.inverse_transform(Y_test),
    "pred_cell_type": label_encoder.inverse_transform(Y_test_hat),
    "easy_max_prob": model.predict_proba(X_test).max(axis=1)
})

easy_pred_df.to_csv("predictions_easy_cell_type.csv", index=False)
print("Saved:", "predictions_easy_cell_type.csv")
easy_pred_df.head()


# In[21]:


# Train the ml model in a more difficult task
# Use 'author_cell_type' instead of 'cell_type'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Determine cell type column (update this if different)
celltype_column = 'author_cell_type'  # Adjust based on your dataset

# Extract features (gene expression matrix)
X = data.X
if hasattr(X, 'toarray'):
    X = X.toarray()  # Convert sparse to dense

print(f"Feature matrix (X) shape: {X.shape}")
print(f"Type: {type(X)}")

# Extract labels (cell types)
Y_raw = data.obs[celltype_column].values
print(f"\nLabel array (Y) shape: {Y_raw.shape}")
print(f"Unique cell types: {np.unique(Y_raw)}")
print(f"Number of cell types: {len(np.unique(Y_raw))}")

# Encode labels as integers (sklearn requires this for some algorithms)
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y_raw)

print(f"\nLabel encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} -> {i}")


# In[22]:


# Subset the data in Train and Test
# Split data: 70% train, 30% test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42,
    stratify=Y  # Maintain class proportions
)

print("\n" + "=" * 50)
print("TRAIN/TEST SPLIT")
print("=" * 50)
print(f"Total samples: {X.shape[0]}")
print(f"Training samples: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Test samples: {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

# Check class distribution
print(f"\nClass distribution in training set:")
for i, label in enumerate(label_encoder.classes_):
    count = (Y_train == i).sum()
    print(f"  {label}: {count} ({count/len(Y_train)*100:.1f}%)")

print(f"\nClass distribution in test set:")
for i, label in enumerate(label_encoder.classes_):
    count = (Y_test == i).sum()
    print(f"  {label}: {count} ({count/len(Y_test)*100:.1f}%)")


# In[23]:


# Training Logistic Regression 
from sklearn.linear_model import LogisticRegression

print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

# Design the model
model = LogisticRegression(
    max_iter=1000,      # Maximum iterations for convergence
    random_state=42,    # For reproducibility
    verbose=1,          # Show training progress
    n_jobs=-1           # Use all CPU cores
)

print("Training logistic regression model...")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(label_encoder.classes_)}")

# Train the model
model.fit(X_train, Y_train)

print("\n✓ Model training complete!")
print(f"Converged: {model.n_iter_}")
print(f"Model coefficients shape: {model.coef_.shape}")


# In[24]:


# Checking the model parameters
# Examine learned weights
print("\n" + "=" * 50)
print("MODEL PARAMETERS")
print("=" * 50)

print(f"Number of classes: {len(model.classes_)}")
print(f"Classes: {model.classes_}")
print(f"\nCoefficient matrix shape: {model.coef_.shape}")
print(f"  (rows = classes, columns = genes)")
print(f"\nIntercept shape: {model.intercept_.shape}")

# Find most important genes for each class (top weights)
for i, class_label in enumerate(label_encoder.classes_):
    weights = model.coef_[i]
    top_gene_indices = np.argsort(np.abs(weights))[-5:][::-1]
    
    print(f"\nTop 5 genes for predicting '{class_label}':")
    for rank, gene_idx in enumerate(top_gene_indices, 1):
        gene_name = data.var_names[gene_idx]
        weight = weights[gene_idx]
        print(f"  {rank}. {gene_name}: weight = {weight:.4f}")


# In[25]:


# Assessing perfomance on the test subset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

# Predict on training subset (Where we fit the model: So seen data)
Y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, Y_train_pred)

# Predict on test subset (Unseen data)
Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print(f"Harder Task - Seen - > Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Harder Task - Unseen -> Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check for overfitting
if train_accuracy - test_accuracy > 0.1:
    print("\n⚠ Warning: Large gap between train and test accuracy suggests overfitting!")
else:
    print("\n✓ Model generalizes well to unseen data")


# In[26]:


### Test subset evaluation + Detailed per class evaluation + ROC / AUC Visuals (as Git instructions: Another way to see them)
# Evaluate on the test set (predictions + overall accuracy)
import sklearn.metrics as skm


# Predict labels on test set (these are encoded integers)
Y_test_hat = model.predict(X_test)

# Decode to original string labels (author_cell_type names)
Y_test_lbl = label_encoder.inverse_transform(Y_test)
Y_test_hat_lbl = label_encoder.inverse_transform(Y_test_hat)

print("Test accuracy (overall):", skm.accuracy_score(Y_test_lbl, Y_test_hat_lbl))


# In[27]:


# More detailed evaluation: per-class accuracy 
eval = []

for ct in set(Y_test_lbl):
    idx = [i for i, cx in enumerate(Y_test_lbl) if cx == ct]
    res = pd.DataFrame({
        "cell_type": ct,
        "accuracy": sum([Y_test_hat_lbl[i] == ct for i in idx]) / len(idx),
        "number": len(idx)
    }, index=[0])
    eval.append(res)

eval = pd.concat(eval, ignore_index=True)

print(eval.sort_values("accuracy"))


# In[28]:


# Visuals -> Scatter Plot: accuracy (x) vs cell_type (y)

plt.figure(figsize=(6, max(4, 0.25 * eval.shape[0])))
plt.scatter(eval.accuracy, eval.cell_type)
plt.xlabel("Accuracy")
plt.ylabel("Cell type")
plt.title("Per-class accuracy on test set")
plt.show()


# In[29]:


# Visuals -> Scatter plot: accuracy vs number of cells + Pearson correlation

from scipy import stats

plt.figure(figsize=(6, 4))
plt.scatter(eval.accuracy, eval.number)
plt.xlabel("Accuracy (per class)")
plt.ylabel("Number of cells (test set)")
plt.title("Accuracy vs number of cells")
plt.show()

r, p = stats.pearsonr(eval.accuracy, eval.number)
print(f"Pearson r = {r:.3f}, p = {p:.3g}")


# In[30]:


# Classification report (precision/recall/f1 per class)
print(skm.classification_report(Y_test_lbl, Y_test_hat_lbl, digits=3))


# In[31]:


# Visuals -> ROC AUC evaluation: ROC curve per cell type + AUC

import matplotlib.pyplot as plt

Y_test_hat_probs = model.predict_proba(X_test)

fig, axs = plt.subplots(5, 3, constrained_layout=True, figsize=(12, 12))

# Note: professor asked for 15 slots; if you have >15 classes, this will plot first 15.
for i, cl in enumerate(model.classes_[:15]):
    cl_name = label_encoder.inverse_transform([cl])[0]

    fpr, tpr, thresholds = skm.roc_curve(
        [0 + (x == cl_name) for x in Y_test_lbl],
        Y_test_hat_probs[:, i]
    )

    ax = axs.flatten()[i]
    ax.plot(fpr, tpr)
    ax.set_title(cl_name, fontdict={'color':'blue','size':7})

    print(cl_name, skm.auc(fpr, tpr))

# Turn off any unused axes (if <15 classes)
for j in range(len(model.classes_[:15]), 15):
    axs.flatten()[j].axis("off")

plt.show()


# In[32]:


# Save predictions harder model
row_id = np.arange(len(Y_test))

Y_test_hat = model.predict(X_test)

hard_pred_df = pd.DataFrame({
    "row_id": row_id,
    "true_author_cell_type": label_encoder.inverse_transform(Y_test),
    "pred_author_cell_type": label_encoder.inverse_transform(Y_test_hat),
    "hard_max_prob": model.predict_proba(X_test).max(axis=1)
})

hard_pred_df.to_csv("predictions_hard_author_cell_type.csv", index=False)
print("Saved:", "predictions_hard_author_cell_type.csv")
hard_pred_df.head()


# In[33]:


# Merge EASY + HARD predictions into one table

merged_pred_df = easy_pred_df.merge(hard_pred_df, on="row_id", how="inner")

merged_pred_df.to_csv("predictions_easy_and_hard_merged.csv", index=False)
print("Saved:", "predictions_easy_and_hard_merged.csv")
merged_pred_df.head()

