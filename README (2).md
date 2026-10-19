# ⚙️ Machine Guardian AI
## Predictive Maintenance: Predicting Machine Failure with Machine Learning

### 1. Project idea and overview

**Machine Guardian AI** is a beginner-friendly machine-learning project that explores whether sensor measurements from industrial equipment can help predict a machine failure.

Factories use machines for repeated operations, and unexpected equipment failures can interrupt production, delay orders, increase repair costs, and create safety concerns. Maintenance is often performed after a machine breaks down or according to a fixed schedule. Predictive maintenance takes a more data-driven approach: it studies patterns in equipment measurements to help identify records that may be associated with a failure.

In this project, we use measurements such as air temperature, process temperature, rotational speed, torque, tool wear, and product type. The model learns patterns from historical examples where the outcome is already known. After training, it receives an unseen set of measurements and predicts whether that record belongs to the **machine-failure** or **no-machine-failure** class.

This is a **binary classification** problem:
- `0` — no machine failure recorded
- `1` — machine failure recorded

The prediction is an early-warning signal for learning and experimentation, not a guarantee that a machine will or will not fail. The dataset is a synthetic benchmark designed to reflect industrial predictive-maintenance data, so a real factory would need its own representative data and safety validation. The dataset contains 10,000 records and 14 columns according to the UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

### 2. Model names and what they predict

This project builds and compares two beginner-friendly classification models:

#### Model A — Decision Tree Classifier
A Decision Tree learns a sequence of feature-based rules. For example, it may learn that certain combinations of torque, rotational speed, temperature, and tool wear are associated with failure records. It is relatively easy to explain because its decisions follow a tree of splits.

**Prediction:** whether a machine record is classified as `0` (no failure) or `1` (failure).

#### Model B — K-Nearest Neighbors (KNN) Classifier
KNN compares a new record with similar records in the training dataset. It looks at the nearest examples in feature space and predicts a class based on their labels. Because features use different units and ranges, the notebook scales numeric measurements before using KNN.

**Prediction:** whether the new machine record is classified as `0` (no failure) or `1` (failure), based on similar training examples.

The notebook compares both models using test accuracy, classification reports, and confusion matrices. It does not assume in advance which model will win; the result is calculated when the notebook is run.

### 3. Why is this project useful?

- **Reduce unplanned downtime:** a potential warning can help maintenance staff decide which equipment records need closer inspection.
- **Plan maintenance more thoughtfully:** teams may be able to investigate warning patterns before relying only on fixed schedules.
- **Understand industrial AI:** the project shows how data science can be applied to manufacturing rather than only to familiar classroom datasets.
- **Learn model evaluation:** the confusion matrix helps distinguish correct predictions from false alarms and missed failure cases.
- **Build practical ML skills:** it demonstrates dataset loading, data-quality checks, feature preparation, training/testing, classification, and result interpretation.

A high accuracy score alone is not enough for a real maintenance system. This dataset has many more non-failure records than failure records, so the notebook also reports precision and recall for the failure class. A missed failure (false negative) may be especially important in an industrial setting.

### 4. Dataset and valid CSV file

**Dataset name:** AI4I 2020 Predictive Maintenance Dataset  
**Source:** UCI Machine Learning Repository  
**Official dataset page:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

**Direct CSV file:** [Open or download the dataset CSV](https://raw.githubusercontent.com/somayaahmed27/AI4I-2020-Predictive-Maintenance-Dataset/main/AI4I%202020%20Predictive%20Maintenance%20Dataset.csv)

The direct link points to a `.csv` file, which can be downloaded and opened in Microsoft Excel. If the browser displays raw text, use the download option or right-click the link and choose **Save link as**. The CSV has 10,000 data records and a header row.

Important columns include:
- `Type`: product type (categorical)
- `Air temperature [K]`: air temperature in kelvin
- `Process temperature [K]`: process temperature in kelvin
- `Rotational speed [rpm]`: rotational speed in revolutions per minute
- `Torque [Nm]`: torque in newton-metres
- `Tool wear [min]`: tool wear measurement in minutes
- `Machine failure`: target label (0 or 1)

The dataset also contains identifier fields and failure-subtype columns. The notebook removes identifiers and the subtype columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` from the input features to avoid target leakage. These subtype indicators are closely related to the overall machine-failure target, so using them as inputs could let a model access information that would not be available in a fair prediction setting.

### 5. Data preparation: loading, missing values, features, and target

The notebook follows these data-preparation steps:

1. **Load the CSV:** download the official dataset archive and read its CSV into a pandas DataFrame.
2. **Inspect the data:** display the first few rows, dataset dimensions, column names, and data types to understand what each column contains.
3. **Check missing values:** count missing values in every column. The UCI description reports no missing values for this dataset, but the notebook still performs the check rather than assuming the data is clean.
4. **Check duplicate records:** count duplicate rows so possible repeated records can be noticed.
5. **Separate features and target:** `Machine failure` is the target (`y`) that the model tries to predict. The remaining appropriate sensor measurements and product type are the input features (`X`).
6. **Remove identifier and leakage columns:** `UDI` and `Product ID` are identifiers rather than useful sensor measurements. `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded because they describe failure modes related to the target.
7. **Prepare data types:** numeric measurements are handled as numeric values, while `Type` is treated as a categorical feature and one-hot encoded.
8. **Handle missing values safely:** a preprocessing pipeline uses the median for numeric columns and the most frequent category for categorical columns if missing values are present.
9. **Split the data:** use a stratified 80% training and 20% testing split so both subsets retain a similar proportion of the rare failure class.
10. **Scale for KNN:** standardize numeric values for KNN, because distance calculations can be dominated by variables with larger numeric ranges. The Decision Tree pipeline does not need numeric scaling.

The preprocessing steps are placed inside scikit-learn pipelines so that values such as medians and scaling parameters are learned from the training set rather than the test set.

### 6. Model building

The notebook trains:
- `DecisionTreeClassifier` with a limited tree depth and balanced class weights.
- `KNeighborsClassifier` with `k = 7` and distance-based voting.

The same training/test split is used for both models, making the comparison fairer. The random seed is fixed for reproducibility.

### 7. Evaluation and results

After training, both models make predictions on the unseen test set. The notebook displays:

- **Accuracy:** the overall proportion of correct predictions.
- **Precision for failure:** of the records predicted as failures, how many were actual failures.
- **Recall for failure:** of the actual failure records, how many the model detected.
- **F1-score:** a combined measure of precision and recall.
- **Confusion matrix:** counts of true negatives, false positives, false negatives, and true positives.

The notebook also draws a bar chart comparing test accuracy. **Actual scores are generated when you run the notebook.** They are intentionally not written as fixed numbers in this README because the reported results should match the executed code and dataset.

When interpreting the result, do not select a model using accuracy alone. Since failure records are uncommon, a model can achieve high accuracy by predicting “no failure” most of the time while missing many failures. Consider failure-class recall and the confusion matrix alongside accuracy.

### 8. Setup instructions

**Requirements**
- Python 3.10 or newer recommended
- Jupyter Notebook, JupyterLab, or Google Colab
- Internet access for downloading the dataset
- Python libraries: pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and requests

**Option A — Run locally with Jupyter**
1. Install Python if it is not already installed.
2. Download this repository to your computer or use **Code → Download ZIP** on GitHub.
3. Open a terminal in the project folder.
4. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn requests jupyter
   ```

5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `Machine_Guardian_AI_Predictive_Maintenance.ipynb`.
7. Run the cells from top to bottom. The data-loading cell downloads the dataset ZIP from UCI and reads the CSV automatically.

**Option B — Run in Google Colab**
1. Upload `Machine_Guardian_AI_Predictive_Maintenance.ipynb` to Google Colab, or open the notebook from your GitHub repository.
2. If needed, install missing libraries in a notebook cell using `!pip install pandas numpy matplotlib seaborn scikit-learn requests`.
3. Run all cells in order and allow the dataset download cell to finish.

### 9. Conclusion

Machine Guardian AI demonstrates how two simple machine-learning algorithms can use industrial sensor measurements to classify machine records as failure or no failure. The Decision Tree offers rule-based decisions that are comparatively easy to explain, while KNN bases its prediction on similar examples in the training data.

After running the notebook, write the conclusion using the measured results:
- Which model achieved the higher test accuracy?
- Which model had better recall for the failure class?
- What do the confusion matrices reveal about false alarms and missed failures?
- Why might the model with the highest accuracy not always be the best maintenance model?
- What limitations should be addressed before real-world use?

**Final learning:** the project connects data preparation, classification, and evaluation to a practical industrial problem. It also shows why a responsible ML project must discuss class imbalance, data leakage, and the limits of its predictions—not just report an accuracy score.

### 10. Limitations and future improvements

- The dataset is synthetic, not a direct feed from a real factory.
- Failure events are relatively rare, so accuracy may give an incomplete picture.
- A random train/test split may not reflect how a model performs on future time periods or different machines.
- A real system would need validation on representative operational data and review by maintenance and safety experts.
- Future work could compare additional models, use cross-validation, tune decision thresholds, and investigate feature importance.

### References
1. Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. International Conference on Artificial Intelligence for Industries.
2. UCI Machine Learning Repository. *AI4I 2020 Predictive Maintenance Dataset*. https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
3. Dataset CSV mirror: [AI4I 2020 Predictive Maintenance Dataset.csv](https://raw.githubusercontent.com/somayaahmed27/AI4I-2020-Predictive-Maintenance-Dataset/main/AI4I%202020%20Predictive%20Maintenance%20Dataset.csv)
