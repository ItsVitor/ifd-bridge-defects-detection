# Experimental Pipeline Specification

## 1. Data Loading

- Iterate through all 7 Parquet files defined in `data.md`:
  - **Grouping:** Process data per **Sample** (defined as a unique `ExperimentID`).
- Consolidate everything into a single dataframe.

## Preprocessing Phase

### 2. Feature Extraction

**Note:** Not every pipeline will have the feature extraction step during it's preprocessing phase, since there are models which utilize the raw input and does automatic feature extraction during their execution, such as autoencoders.

**Feature Extraction Scope:** Features (RMS, Kurtosis, etc.) must be calculated per sensor `NodeID` (18 sensors per Experiment).

**Normalization Rule:**

- **Do NOT scale** the raw `Accel_` columns (this preserves physical amplitude differences) (although I would think that since the acceleration measures already come normalized from -1 to 1, this specification might not be necessary).

**Objective:** Transform raw time-series data into a structured feature matrix $X$.

1. **Calculation:**
    - For each Sample (experiment), process all 18 sensor nodes.
    - For each node, process all 3 axes (`Accel_X`, `Accel_Y`, `Accel_Z`) independently.
    - **Time-Domain Features (per axis per node):** Calculate Mean, Covariance (Variance), Kurtosis, RMS, Peak-to-RMS, RSS, Peak-to-Peak, Minimum, Maximum, and Feature-jerk.
    - **Frequency-Domain Features (per axis per node):** Calculate PSD (using $f_s=256$ Hz), then extract Spectral Centroid, Spread, Skewness, Kurtosis, and Entropy.
2. **Output Structure:**
    - Create a Master Feature Matrix where:
      - **Rows** = Experiments (one row per ExperimentID)
      - **Columns** = 810 features (18 nodes $\times$ 3 axes $\times$ 15 features)
      - **Feature Naming**: `{axis}_N{node_id}_{feature}` (e.g., `X_N1_mean`, `Z_N18_entropy`)
    - **Metadata Columns:** `Group_ID` (EOV combination) and `Class` (0=Healthy/Slightly Damaged, 1=Damaged).

### Autoencoders Output Structure

Even for the pipelines that does not have a feature extraction step, like autoencoders in the present case, the data still needs to be consolidated into a format that can be fed onwards into the training phase.

1. **Signal Matrix:**
    - Each Sample (experiment) becomes one row holding its raw acceleration signals, flattened from shape `(timesteps, channels)`.
    - **Channels** = selected nodes $\times$ selected axes (`nodes_to_use` and `axes_to_use` in `config/autoencoders.yaml`; default 8 nodes $\times$ X, Y = 16 channels).
    - **Column Naming:** `{axis}_N{node_id}_t{i}` (e.g., `X_N10654_t0`).
    - **Metadata Columns:** `Group_ID` and `Class`, same as above.
2. **Signal Length:** Records have 1827, 2010 or 2233 samples depending on train speed. All signals are cropped to the shortest length, rounded down to a multiple of the total pooling factor (1827 → 1824 samples, ≈ 7.1 s). Signals are **not** resampled, since that would shift the modal frequencies the model is meant to learn.
3. **Reshape:** The `normalize` step reshapes each row back to `(timesteps, channels)` before training.



---

## 3. Experimental Strategy (The Outer Loop)

**Objective:** Perform 10-Fold Group Cross-Validation to evaluate model performance.

- **Split Strategy:** Use the pre-defined "Assignment Matrix" (images/Cross-validation folds table.png) to assign specific EOV groups to the **Test Set** for each fold.
Dataset contains 45 EOV groups (combinations of speed, mass, elasticity), this is the 10-fold cross-validation setup:

  - Folds 1-5: 4 EOV groups each;

  - Folds 6-10: 5 EOV groups each.

Each fold's test set uses unique EOV groups not seen in training.

### Step 3.1: Data Partitioning

1. **Test Set:** Select all samples belonging to the EOV groups assigned to the current `Fold_ID`.
2. **Training Set:** Select all samples belonging to the remaining EOV groups.
3. **Strict Balancing:** Ensure the Test Set has exactly a 1:1 ratio of Healthy to Damaged samples (downsample the majority class if necessary).

### Step 3.2: Feature Filtering

1. **Variance Threshold Filter**: Remove features with zero or near-zero variance.
2. **Correlation Filter**: Remove highly correlated features (threshold=0.95) to reduce redundancy.
3. **Fit** both filters using **only** the Training Data.
4. **Transform** both Training and Test Data using the fitted filters.
5. Implemented in `BridgeDefectPipeline.select_features`. The autoencoder pipeline overrides it as a no-op (see Branch C).

### Step 3.3: Data Normalization (Anti-Leakage)

1. Initialize a `StandardScaler`.
2. **Fit** the scaler using **only** the Training Data.
3. **Transform** the Training Data.
4. **Transform** the Test Data using the parameters learned from Training.

---

## 4. Model Training

**Objective:** Train models with literature-based hyperparameters.

### Branch A: Unsupervised Models (OCSVM & Isolation Forest)

- **Training Data:** Use **ONLY** Healthy samples (`Class 0`) from the Training partition.
- **Hyperparameters:** Fixed values based on literature recommendations (no optimization).
- **Training:** Fit model on healthy samples to learn "normality".
- **Prediction:** Predict class labels for the **Outer Test Set**.
  - OCSVM: Inliers (1) → Healthy (0), Outliers (-1) → Damaged (1)
  - Isolation Forest: Inliers (1) → Healthy (0), Outliers (-1) → Damaged (1)

### Branch B: Upper Bound Baseline (Random Forest)

- **Training Data:** Use **BOTH** Healthy (`0`) and Damaged (`1`) samples from the Training partition.
- **Hyperparameters:** Fixed values based on literature recommendations.
- **Training:** Standard supervised learning.
- **Prediction:** Predict class labels for the **Test Set**.

### Branch C: Unsupervised Model (Sparse Conv1D Autoencoder)

- **Input:** Raw signal matrix (see *Autoencoders Output Structure*), not statistical features.
- **Feature Filtering:** Skipped. Columns are time instants, so variance/correlation filters would delete time steps and break the temporal continuity the convolutions rely on. Adjacent samples are naturally highly correlated; that is signal structure, not redundancy.
- **Normalization:** Per-channel standardization (mean 0, std 1) fitted on training data only. Statistics are pooled over samples and time, so the waveform shape is preserved.
- **Training Data:** **ONLY** Healthy samples (`Class 0`) from the Training partition. 10% held out (after shuffling) for early stopping.
- **Architecture:** Hyperparameters in `config/autoencoders.yaml`.
  - Encoder: 3 blocks of `Conv1D(kernel=9, ReLU, padding="same")` + `MaxPooling1D`, filters `[32, 32, 16]`, pools `[4, 4, 2]` (1824 → 57 timesteps, 16 channels → 16 latent channels).
  - Sparse bottleneck: L1 activity regularization (`1e-4`) on the latent representation.
  - Decoder: mirror of the encoder with `UpSampling1D`, ending in a linear `Conv1D` with one filter per input channel.
  - Training: Adam (`lr=1e-3`), MSE loss, up to 100 epochs, batch 32, early stopping (patience 10, best weights restored).
- **Threshold:** $\tau = \mu_{train} + k\,\sigma_{train}$ of the per-sample reconstruction MSE on the healthy training samples (`threshold_k = 1.0`).
- **Prediction:** Reconstruction MSE $> \tau$ → Damaged (1), otherwise Healthy (0).

---

## 5. Evaluation (Post-Processing)

**Objective:** Assess accuracy metrics and statistical significance of the results.

1. **Metric Calculation:** For each fold, calculate **Accuracy**:
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Aggregation:** Compute the Mean and Standard Deviation of Accuracy across the 10 folds.
3. **Statistical Testing:**
    - Compare OCSVM results vs. Random Forest Baseline.
    - Compare iForest results vs. Random Forest Baseline.
    - **Tests:** Perform **Corrected Paired Student's t-test** and **Wilcoxon Signed-Rank**
