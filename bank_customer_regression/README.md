# Linear Regression Folder

Despite the folder name, this contains both advertising regression and bank-customer classification/neural-network experiments.

## Code Files

| File | What it does |
|---|---|
| `advertising_linear_regression.py` | Advertising CSV linear regression with file checks, train/test split, metrics, and residual table. |
| `advertising_linear_regression_variant.py` | Variant/copy of the advertising linear-regression script with nearly identical logic. |
| `bank_customer_lr_ann_comparison.py` | Bank customer churn workflow comparing logistic regression and a Keras neural network with class weights and plots. |
| `bank_customer_ann_fixed.py` | Bank customer logistic regression plus Keras ANN, with warning fixes and training history plotting. |
| `bank_customer_ann_basic.py` | Bank customer logistic regression and neural-network variant. |
| `bank_customer_ann_basic_copy.py` | Exact duplicate/copy of `bank_customer_ann_basic.py`. |

## Data

| File | Purpose |
|---|---|
| `BankCustomer.csv` | Customer churn dataset. |
| `nasdaq100.csv` | Local stock/index dataset. |

Note: the advertising scripts expect `./dataset/advertising.csv`, which is not currently inside this folder.
