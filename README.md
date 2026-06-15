
---
# **🔭 Project Overview**

## 🧩K-Prototype Clustering: Optimized From-Scratch Implementation

A high-performance, from-scratch Python implementation of the K-Prototype clustering algorithm. This repository provides a purely vectorized approach to handling mixed data types (numerical and categorical), bypassing the overhead of external clustering libraries. 

It is specifically designed for complex, real-world datasets—Zomato Bangalore Restaurants dataset—where categorical features often contain multi-valued, unsorted strings.

## ⚙️Core Features & Optimizations

### 1. 🔐 Multi-Token Categorical Encoding
Real-world categorical data often contains delimited strings (e.g., "North Indian, Chinese" vs. "Chinese, North Indian"). 
* **The `in_depth_encoder`** splits, alphabetically sorts, and reconstitutes these strings before encoding. This ensures permutations of the same distinct items are mapped to the exact same categorical integer.
* **$O(1)$ Decoding:** Categorical mappings are stored as a NumPy array, allowing the `in_depth_decoder` to reverse integer codes back to strings instantly via direct index mapping.

### 2. </> Pure NumPy Vectorization
Native Python loops are entirely stripped from the core mathematical operations. 
* Numerical distances and categorical mismatches are computed simultaneously across all prototypes utilizing `np.newaxis` broadcasting. 
* The cost function computes cluster assignments using `np.argmin` and `np.min` directly along the tensor axes.
* **Hardware Efficiency:** By keeping all operations strictly in-memory through NumPy arrays, the algorithm avoids virtual memory paging. This prevents the 100% SSD usage spikes common in heavy data operations, ensuring stable execution on local hardware and laptops (e.g., MSI Cyborg 15 series).

### 3. { } Automated Dead-Cluster Recovery
The `empty_prototype_checker` continuously monitors the `cluster_counts` matrix. If a prototype is orphaned (assigned 0 data points), the algorithm intercepts the matrix and randomly re-initializes that specific prototype from the active dataset pool. This guarantees $K$ clusters are always maintained without throwing runtime exceptions.

### 4. 🧠 Smart Initialization & Automatic Gamma
* **Best-Attempt Initialization:** The `best_init_prototype_points` function runs `n_attempts` of random initializations and pre-calculates the initial cost, selecting the starting centroid positions with the absolute lowest starting loss.
* **Dynamic $\gamma$:** If no gamma weight is provided, `set_gamma` automatically scales the categorical mismatch penalty against the numerical variance by calculating `0.5 * numerical_array.std()`.

### 5. 📊 Integrated Diagnostic Plotting
The `elbow_plotter` automates the iteration over multiple $K$ values, storing convergence costs and generating an elbow plot. The visualizer automatically handles directory creation (`..\results\figures`) and exports high-DPI `.png` files for analysis.

---
# ** Data Preparation Pipeline**

Handling real-world data like the Zomato Bangalore dataset requires strict type casting, outlier rejection, and scale normalization to ensure the K-Prototype algorithm converges efficiently without numerical instability.

## Part 1: 🧹 Data Cleaning

The raw dataset contained unformatted strings, undefined nulls, and irrelevant metadata that needed to be stripped before any mathematical operations could occur.
	
* **🗑️ Dimensionality Reduction via Feature Dropping**
	* *Problem:* Memory bloat from unanalyzable textual and metadata columns.
	* *Solution:* Hard-dropped `url`, `phone`, `address`, `dish_liked`, `reviews_list`, and `menu_item`.
	
* **🔠 String Parsing & Type Casting (Numerical Columns)**
	* *Problem:* Numerical fields were stored as messy strings (e.g., ratings appended with "/5", costs containing commas).
	* *Solution:* Stripped standard formatting artifacts using vectorized `.str.replace()` and coerced the results to standard floats.
	* *Example (Cost):* `"1,500"` $\rightarrow$ `1500.0`
	* *Example (Rate):* `"4.1/5"` $\rightarrow$ `4.1`
	
* **🤞Outlier Rejection & Mean Imputation**
	* *Problem:* Bad data entry resulted in impossible values (e.g., ratings $>5$ or $<0$).
	* *Solution:* Applied logical masking to force out-of-bounds values to `NaN`. All `NaN` fields were then imputed using the exact mean of the cleaned column array.
	* *Example (Rate):* `NaN` or `9.9` $\rightarrow$ `3.7` (Column Mean)
	
* **🗂️ Categorical Null Handling**
	* *Problem:* Missing text fields break categorical encoders.
	* *Solution:* Replaced `NaN` in categorical columns with explicitly defined string flags.
	* *Example (Location):* `NaN` $\rightarrow$ `"No Location"`


## Part 2: 🛠️ Feature Engineering

With the data structurally clean, it required mathematical transformation to optimize the distance and mismatch calculations within the K-Prototype model.
	
* **🧮 Binary Encoding**
	* *Problem:* `online_order` and `book_table` were textual arrays (`"Yes"`/`"No"`).
	* *Solution:* Mapped to binary integers via pure NumPy masking (`np.where`), making them immediately readable by the algorithm.
	* *Example:* `"Yes"` $\rightarrow$ `1` | `"No"` $\rightarrow$ `0`
	
* **🏗️ Logarithmic Normalization of Continuous Variables**
	* *Problem:* Features like `votes` and `approx_cost` possessed massive variance and high skewness, which mathematically overpowers standard Euclidean distance, rendering smaller variance columns like `rate` irrelevant.
	* *Solution:* Applied `np.log1p()` (Log base $e$ of $1 + x$) to compress the scale while safely handling any zero values. Any new `NaN`s generated during the shift were re-imputed with the normalized mean.
	* *Example (Votes):* `775` $\rightarrow$ `6.654153`
	
* **🔍 In-Depth Categorical Encoding**
	* *Problem:* Multi-label strings (e.g., `"Chinese, North Indian"`) are treated as entirely distinct from permutations (e.g., `"North Indian, Chinese"`) by standard encoders.
	* *Solution:* Passed all textual columns through the custom `in_depth_encoder`. This alphabetized and reconstituted multi-token strings before mapping them to dense integers. The decoder reference arrays were immediately exported to `.npy` files to allow rapid integer-to-string mapping in the future.
	* *Example (Cuisines):* `"North Indian, Mughlai, Chinese"` $\rightarrow$ `1436`
	
* **🔢 Matrix Separation & Export**
	* *Problem:* The K-Prototype algorithm expects strict, separate $N \times M$ arrays for numerical and categorical calculations.
	* *Solution:* Split the final encoded DataFrame into `categorical_data_frame` and `numerical_data_frame`, exporting them as isolated, production-ready `.csv` files.

---

Here is the deep dive section for your README. It breaks down the mathematical and programmatic flow of your algorithm, directly referencing your architecture to highlight the engineering choices you made.

---

# 🧠 Logic & Process: A Deep Dive

The K-Prototype algorithm is a hybrid model that marries the Euclidean distance of K-Means (for continuous data) with the matching dissimilarity of K-Modes (for discrete data). This implementation executes that logic through a strictly vectorized pipeline, eliminating bottlenecks typically caused by native Python loops over large datasets.

Here is the step-by-step breakdown of the internal mechanics.

## 1. 🎯 Prototype Initialization & Gamma Calibration
Before the iterative process begins, the algorithm must establish starting parameters.

* **Dynamic Gamma ($\gamma$):** The algorithm inherently requires a scaling factor, $\gamma$, to balance the weight of categorical mismatches against numerical variance. If omitted by the user, the `set_gamma` function dynamically computes this as half the standard deviation of the numerical array:
```python
def set_gamma(self, numerical_array):
    self.gamma = 0.5 * numerical_array.std()
```

* **Best-Attempt Seeding:** Rather than generating synthetic coordinates, `random_prototypes_initializer` slices $K$ actual rows from the dataset. To avoid poor local minima, the `best_init_prototype_points` wrapper runs this initialization `n_attempts` times, evaluates the baseline cost for each, and proceeds with the absolute lowest-cost starting state.

## 2. ⚖️ The Cost Function (Vectorized Broadcasting)
The core of K-Prototype requires computing the distance from every data point to every cluster center. This is handled using NumPy's `np.newaxis` to project the arrays into 3D tensors, computing distances simultaneously.

* **Numerical Cost (Squared Euclidean):**
```python
def numerical_distance(self, numerical_array, numerical_prototypes):
    diff = numerical_array - numerical_prototypes[:, np.newaxis]
    return np.sum(diff**2, axis=-1)
```

* **Categorical Cost (Hamming Mismatch):**
```python
def categorical_mismatch(self, categorical_array, categorical_prototypes):
    mismatches = categorical_array != categorical_prototypes[:, np.newaxis]
    return np.sum(mismatches, axis=-1)
```

* **Total Cost Convergence:** These matrices are flattened into a single cost matrix:
	`total_distance_array = distance_results_array + gamma * mismatch_results_array`

## 3. 🖧 Cluster Assignment
With the `total_distance_array` calculated, assigning points to their nearest prototype is reduced to a single operation.

```python
def data_clusterer(self, total_distance_array):
    distinguisher_labels_array = np.argmin(total_distance_array, axis=0)
    return distinguisher_labels_array
```

This returns a 1D array (e.g., `[0, 2, 1, 1...]`), acting as a mapping mask for the update phase.

## 4. ♻️ Centroid Re-calculation
Once clustered, the prototypes must be shifted to the mathematical center of their assigned points.

* **Numerical Prototypes (Mean):** The algorithm uses a dot product between a boolean `weight_matrix` and the original data array. This mathematically sums the grouped coordinates, which are then divided by the `cluster_counts` to find the exact mean. `np.maximum(cluster_counts, 1)` acts as a safeguard against ZeroDivision errors.
* **Categorical Prototypes (Mode):** Because text/integer encoding cannot be averaged, the algorithm extracts the exact mode (most frequent value) across all columns in the cluster simultaneously using `scipy.stats.mode`.

## 5. ❌ Dead-Cluster Interception
In highly skewed datasets, a prototype may occasionally fail to "capture" any data points, effectively dying. The algorithm prevents this runtime failure through the `empty_prototype_checker`.

```python
empty_prototype_indices = np.where(cluster_counts.flatten() == 0)[0]

```

If an orphaned cluster is detected, it is instantly re-seeded using a random point from the active dataset, ensuring exactly $K$ clusters exist at all times.

## 6. 🤝 Convergence
The `prototype_converger` locks these steps into a `while` loop. The cycle repeats until one of two conditions is met:

1. `max_iterations` is reached (default: 1000).
2. The arrays for `numerical_prototypes` and `categorical_prototypes` exactly match the arrays from the previous iteration (`np.array_equal`), indicating mathematical convergence.

---
# 📊 Model Execution & Results Interpretation

To benchmark the algorithm on the processed Zomato dataset, the model was executed with the following configuration:

```python
numerical_prototypes, categorical_prototypes = K_Model.fit(
    numerical_array=encoded_numerical_features, 
    categorical_array=encoded_categorical_features, 
    K=10, 
    gamma=None, 
    n_attempts_for_best_init_prototypes=350, 
    max_convergance_iterations=500, 
    timer=True
)
```

## 1. 📝 Hyperparameter Context

* **`K=10`**: The model was tasked with finding 10 distinct restaurant "archetypes" within the Bangalore market.
* **`gamma=None`**: The model auto-calibrated the categorical mismatch penalty against the standard deviation of the scaled numerical arrays (Rates, Votes, Cost).
* **`n_attempts_for_best_init_prototypes=350`**: This is a highly aggressive search strategy. Before the first actual iteration, the algorithm blindly sampled 350 combinations of 10 random centroids, computed the cost function for the entire dataset 350 times, and only proceeded with the specific starting grid that yielded the absolute lowest initial mathematical loss. This effectively bypasses the risk of bad local minima.
* **`max_convergence_iterations=500`**: Provided a ceiling to stop the algorithm if absolute convergence (`array_equal`) took an unreasonable amount of time.

## 2. 💡Interpreting the Discovered Prototypes (Cluster Centers)

The output CSV (`prototypes_data_frame.csv`) reveals that the K-Prototype algorithm successfully segmented the Bangalore restaurant scene into distinct, highly logical market profiles. Because this algorithm returns actual categorical modes rather than fractional averages (like standard K-Means would if forced), the prototypes represent readable "personas."

Here is the breakdown of the major clusters discovered:

#### 🟢 The Elite / High-Demand Hubs (Cluster 4)
	
* **Profile:** High Cost (₹1136), Exceptionally High Votes (~1788), High Rating (4.26).
* **Traits:** Casual Dining, North Indian, located in *Koramangala 5th Block*.
* **Insight:** These are the established, premium heavy-hitters. They support both online ordering and table booking, indicating high infrastructure and massive foot/digital traffic.

#### 🍸 Premium Nightlife & Experiences (Cluster 3)
	
* **Profile:** Highest Cost (₹1372), Good Rating (~4.00), Moderate Votes (~280).
* **Traits:** Finger Food, *MG Road*, Table Booking available.
* **Insight:** The algorithm successfully isolated premium pubs and lounges. Noticeably, this is the only major cluster that **does not** accept online orders—which mathematically aligns perfectly with "Finger Food/Pubs" where the value is in the physical dine-out experience rather than home delivery.

#### 🟡 Mid-Tier Casual Dining (Clusters 0 & 1)
	
* **Profile:** Medium Cost (₹550–₹650), Good Ratings (3.80–4.00), Solid Votes (240–650).
* **Traits:** Located in prime IT/Young-professional hubs (*Indiranagar* and *Koramangala*).
* **Insight:** The "bread and butter" of Bangalore's food scene. These places rely heavily on online orders but do not require table bookings.

#### 🔴 The BTM "Quick Bite" Ecosystem (Clusters 2, 7, 8, & 9)
	
* **Profile:** Low Cost (₹310–₹350), Low Ratings (3.30–3.70), Very Low Votes (0–100).
* **Traits:** *BTM Layout*, Quick Bites, Chinese/North Indian.
* **Insight:** The algorithm grouped the massive density of budget eateries and cloud kitchens in BTM Layout. Interestingly, the model split them further based on their digital footprint: Cluster 2 and 7 ignore online ordering (likely purely local street food or canteens), whereas Clusters 8 and 9 are heavily integrated into delivery apps.

---
## ✅ Conclusion of Results

The algorithm successfully preserved the nuance between numerical weight (cost vs. votes) and categorical restrictions. It accurately deduced that highly expensive MG Road places don't do online delivery, while low-cost BTM Layout joints survive almost entirely on the delivery ecosystem—proving the viability and optimization of this from-scratch implementation.