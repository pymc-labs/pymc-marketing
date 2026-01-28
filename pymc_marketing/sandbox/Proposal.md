
## Technical Specification: MMM Data & Plotting Framework

### 1. Problem Statement & Requirements

The current implementation in `PyMC-Marketing` lacks a formal data-handling layer and interactive plotting suite, leading to several structural issues that hinder both development and user experience.

#### A. The "Trapped Data" Problem

* **Requirement:** For every plot generated, users must have access to the underlying summary statistics (e.g., a DataFrame) used to create that plot.
* **Current Issue:** Existing plotting functions perform data wrangling internally. This "hides" the data from the user. Even when using Arviz, summary statistics are computed internally and not exposed. We need a way to compute and provide these statistics independently.

#### B. Rigidity of `InferenceData` (iData)

* **Requirement:** We need a codified way to work with `iData` that supports easy manipulation.
* **Specific Needs:**
* **Time Aggregation:** Ability to easily roll up data to monthly or yearly grains (crucial for ROI/ROAS reporting).
* **Field Operations:** Standardized methods to filter fields or aggregate them (e.g., combining multiple social channels into a single "Social" category).



#### C. Complex Visualization Requirements

* **Requirement:** Support interactive plots (Plotly) and models with arbitrary dimensions (Date, Channel, Brand, etc.).
* **Use Case Complexity:**
* **Multi-Dimensionality:** A model with 3 channels and 2 brands must be able to display all 6 saturation curves at once, filter down to specific brands.
* **Layout Logic:** The system must support grouping by dimensions to create subplots (e.g., one subplot per brand) or grouped bar charts (e.g., comparing the same channel across different brands).



---

### 2. Proposed Three-Component Architecture

To meet the requirements above, we propose moving away from monolithic plotting functions toward a decoupled, three-tier system:

#### Component 1: Codified Data Wrapper

A "stock" shareable object that wraps the raw `iData`. It provides a consistent API for:

* Time-based resampling (Monthly/Yearly).
* Dimension filtering and grouping (e.g., filtering by brand or aggregating by channel).
* Codifying conventions so the object is "aware" of MMM-specific structures.

#### Component 2: MMM Summary Object

A transformation layer that takes the Codified Data and produces a structured `DataFrame` of summary statistics.

* **Role:** This acts as the "Source of Truth" for both the user and the plotting engine.
* **Output:** Tables containing means, medians, and Credible Intervals (HDI).

#### Component 3: Plotting Suite

A set of functions that consume the **MMM Summary Object** to generate Plotly figures.

* **Logic:** Since the summary object is already filtered or aggregated, the plotting suite remains "thin" and focused purely on visualization and layout.

---

To move toward a final solution, we need to answer the following:

1. Does our solution answers the requirements?
2. Provide feedback on the solution and suggest improvements
3. If there is a better solution provide it.
