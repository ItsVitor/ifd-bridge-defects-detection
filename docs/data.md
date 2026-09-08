# Project Data Specification

## 1. File Inventory

The dataset consists of **7 Parquet files** located in the `/data` folder.

Notes:

- In `/data/v1` there are the original simulated data in 2025;
- In `/data/v2` there are the newly simulated data in 2026 after fixing some things in the numerical model of the bridge.

These are the files:

- **Healthy Class (Training & Testing):**
  - `healthy.parquet` (Strictly healthy data)
  - `slightly_damaged.parquet` (Baseline variations, also **labeled as Healthy/Class 0**)
- **Damaged Class (Testing Only):**
  - `damaged_d1.parquet` (Damage Type 1)
  - `damaged_d2.parquet` (Damage Type 2)
  - `damaged_d3.parquet` (Damage Type 3)
  - `damaged_d4.parquet` (Damage Type 4)
  - `damaged_d5.parquet` (Damage Type 5)

## 2. Data Schema & Column Definitions

All files share a common schema, except for the **Damaged** files which contain one additional target column (`Dano_Percentual`).

### Common Columns (Present in ALL files)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `ExperimentID` | `int` | Unique identifier for a single simulation run. |
| `Velocidade` | `int` | **EOV:** Train speed (e.g., 45, 50, 55 km/h). |
| `Peso_Vagao` | `float` | **EOV:** Wagon mass percentage (e.g., 1.0 = 100%, 0.95 = 95%). |
| `Modulo_Elasticidade` | `float` | **EOV:** Young's Modulus percentage (e.g., 0.975, 1.025). |
| `Irregularidade` | `int` | **EOV:** Track profile ID. **Note:** Healthy files contain IDs 1 & 2; Damaged files contain ONLY ID 1. |
| `NodeID` | `int` | Identifier for the specific sensor. **18 unique NodeIDs**. |
| `Time` | `float` | Timestamp. **Sampling Rate is 256 Hz** ($\Delta t \approx 0.003906s$). |
| `Accel_X` | `float` | Raw Acceleration in X-axis (ranging from -1 to 1). |
| `Accel_Y` | `float` | Raw Acceleration in Y-axis (ranging from -1 to 1). |
| `Accel_Z` | `float` | Raw Acceleration in Z-axis (ranging from -1 to 1). |

### Specific Columns (Damaged files only)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Dano_Percentual` | `float` | Severity of damage (0.01, 0.05, 0.1, 0.2, 0.5, 1.0). |

## 3. Class Labeling

- `healthy.parquet` + `slightly_damaged.parquet` $\rightarrow$ **Class 0 (Normal)**.
- All `damaged_*.parquet` files $\rightarrow$ **Class 1 (Anomaly)**.
