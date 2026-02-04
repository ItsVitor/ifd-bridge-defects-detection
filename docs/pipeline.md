# Experimental Pipeline Specification

## 1. Feature Extraction (Preprocessing Phase)

**Objective:** Transform raw time-series data into a structured feature matrix $X$.

1. **Data Loading:**
    * Iterate through all 7 Parquet files defined in `data.md`.
    * **Grouping:** Process data per **Sample** (defined as a unique `ExperimentID`).
2. **Calculation:**
    * For each Sample (experiment), process all 18 sensor nodes.
    * For each node, process all 3 axes (`Accel_X`, `Accel_Y`, `Accel_Z`) independently.
    * **Time-Domain Features (per axis per node):** Calculate Mean, Covariance (Variance), Kurtosis, RMS, Peak-to-RMS, RSS, Peak-to-Peak, Minimum, Maximum, and Feature-jerk.
    * **Frequency-Domain Features (per axis per node):** Calculate PSD (using $f_s=256$ Hz), then extract Spectral Centroid, Spread, Skewness, Kurtosis, and Entropy.
3. **Output Structure:**
    * Create a Master Feature Matrix where:
      * **Rows** = Experiments (one row per ExperimentID)
      * **Columns** = 810 features (18 nodes $\times$ 3 axes $\times$ 15 features)
      * **Feature Naming**: `{axis}_N{node_id}_{feature}` (e.g., `X_N1_mean`, `Z_N18_entropy`)
    * **Metadata Columns:** `Group_ID` (EOV combination) and `Class` (0=Healthy/Slightly Damaged, 1=Damaged).

---

## 2. Experimental Strategy (The Outer Loop)

**Objective:** Perform 10-Fold Group Cross-Validation to evaluate model performance.

* **Split Strategy:** Use the pre-defined "Assignment Matrix" (Table 4 in the article) to assign specific EOV groups to the **Test Set** for each fold.
* **Loop:** Iterate `Fold_ID` from 1 to 10.

### Inside Each Fold

#### Step 2.1: Data Partitioning

1. **Test Set:** Select all samples belonging to the EOV groups assigned to the current `Fold_ID`.
2. **Training Set:** Select all samples belonging to the remaining EOV groups.
3. **Strict Balancing:** Ensure the Test Set has exactly a 1:1 ratio of Healthy to Damaged samples (downsample the majority class if necessary).

#### Step 2.2: Data Normalization (Anti-Leakage)

1. Initialize a `StandardScaler`.
2. **Fit** the scaler using **only** the Training Data.
3. **Transform** the Training Data.
4. **Transform** the Test Data using the parameters learned from Training.

---

## 3. Model Training & Selection (The Inner Loop)

**Objective:** Optimize Hyperparameters and Features.

### Branch A: Unsupervised Models (OCSVM & Isolation Forest)

* **Optimization Split (Group Shuffle Split):**
  * Partition the **Training Set Pool** into 80% Inner Train and 20% Inner Validation.
  * **Inner Train Set:** Keep **ONLY** Healthy samples (`Class 0`). The model must effectively learn "Normality" from this set.
  * **Inner Validation Set:** Keep **BOTH** Healthy (`Class 0`) and Damaged (`Class 1`) samples. This serves to calibrate the discrimination capability of the selected features.
* **Optimization Routine (Grid Search + SFS):**
    1. **Grid Search:** Iterate through every hyperparameter combination.
    2. **Feature Selection (Wrapper):**
        * **Train:** Fit OCSVM/iForest on the **Inner Train (Healthy Only)**.
        * **Evaluate:** Predict on **Inner Validation (Healthy + Damaged)**.
        * **Metric:** Maximize Accuracy.
    3. **Selection:** Identify the best configuration based on the Validation metric.
* **Final Training:** Retrain the optimal model configuration on the **Full Healthy Training Pool** (excluding all damaged data).
* **Prediction:** Predict class labels for the **Outer Test Set**.

### Branch B: Upper Bound Baseline (Random Forest)

* **Training Data:** Use **BOTH** Healthy (`0`) and Damaged (`1`) samples from the Training partition.
* **Optimization:** Perform the same Nested Grid Search + SFS routine as above
* **Prediction:** Predict class labels for the **Test Set**.

---

## 4. Evaluation (Post-Processing)

**Objective:** Assess accuracy metrics and statistical significance of the results.

1. **Metric Calculation:** For each fold, calculate **Accuracy**:
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Aggregation:** Compute the Mean and Standard Deviation of Accuracy across the 10 folds.
3. **Statistical Testing:**
    * Compare OCSVM results vs. Random Forest Baseline.
    * Compare iForest results vs. Random Forest Baseline.
    * **Tests:** Perform **Corrected Paired Student's t-test** and **Wilcoxon Signed-Rank**
