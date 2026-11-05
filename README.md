# Customer, Product & Profitability Performance Analysis
**APL Logistics (KWE Group) | Unified Mentor project**

## Overview
Order and sales data often hides where a logistics business actually makes money. High sales do not always mean high profit, and discounts, shipping costs and weak customers or categories can quietly erode margins. This project analyses profitability by customer, product, category, market and region, and tests how discounting affects margin, so that pricing and customer decisions are not "profit-blind".

## Dataset
- File: `APL_Logistics.csv` (180,519 rows, 40 columns; read with `encoding="latin-1"`)
- One row is one order item. Covers customers, products, categories, markets, regions, discounts, shipping and profit.
- **No order date column exists**, so time-based margin trends are not possible. Order status and shipping mode are used instead.

## Project structure
```
.
├── APL_Logistics.csv                           # raw data
├── APL_Logistics_Profitability_Analysis.ipynb  # cleaning, feature engineering, analysis, KPIs
├── feature_importance.csv                      # driver analysis (see caveat below)
├── requirements.txt
├── README.md
└── app.py                                      # Streamlit dashboard (add when built)
```

## Method
1. **Cleaning and validation:** removed 4 columns that exactly duplicate others and 4 personal or unused columns; confirmed `Sales = Price x Quantity` and `Order Item Total = Sales - Discount`; found no zero or negative sales, quantity or price. Negative-profit orders were kept on purpose.
2. **Order status rule:** `CANCELED` and `SUSPECTED_FRAUD` orders were excluded from profit analysis because they do not represent realised revenue.
3. **Margin definition:** `Profit / Order Item Total` (net of discount), which matches the dataset's own `Order Item Profit Ratio`. A gross version (`Profit / Sales`) is also reported.
4. **Feature engineering:** margin %, loss flag, discount flag and bands (0, 1-5, 6-10, 11-15, 16-20, 20%+), shipping delay, late flag, customer value index and quartile value tiers.
5. **Analysis:** profit concentration (Pareto), product and category margins, customer tiers, discount diagnostics (Spearman, Welch t-test, Kruskal-Wallis, what-if cap), market, region and country comparison.

## KPIs
| KPI | Definition |
|---|---|
| Total Revenue | Sum of `Sales` |
| Total Profit | Sum of `Order Profit Per Order` |
| Profit Margin % | Profit / Order Item Total (gross version: Profit / Sales) |
| Customer Value Index | Customer profit / average profit per customer |
| Category Margin | Profit / net revenue per category |
| Discount Impact Ratio | (Margin at 0% discount - margin on discounted orders) / margin at 0% discount |

## Key results
- Total sales about 35.2M, total profit about 3.81M, net margin 12.03% (gross 10.81%).
- About 18.7% of orders and 19.8% of customers are loss-making.
- Profit is concentrated in products: 8 of 118 products generate 80% of positive profit (about 31% of customers do the same).
- No category is loss-making and only 2 products are. Category margins range from 0.5% to 19.3%.
- **Discounts show no measurable effect on margin in this data:** margin is about 12% in every discount band (Spearman rho = -0.0016, p = 0.51). The Discount Impact Ratio is 7.6% and is descriptive only.
- What-if discount caps assume unchanged volume and are upper bounds, not forecasts.

## Feature importance (`feature_importance.csv`)
Random Forest models predict `Order Item Profit Ratio` (regression) and loss-making orders (classification) from discount, quantity, price, shipping, type, segment, category, department, market and region. Columns:

| Column | Meaning |
|---|---|
| `rank` | Rank by permutation importance on the margin model |
| `impurity_importance_margin_model` | Built-in Random Forest importance (biased towards high-cardinality features) |
| `permutation_importance_margin_model` | Drop in R² on held-out data when the feature is shuffled |
| `permutation_importance_margin_std` | Spread over 5 repeats |
| `permutation_importance_loss_model_auc` | Drop in AUC for the loss classifier |

**Caveat:** the models have essentially no predictive power on held-out data (test R² = -0.002, test AUC = 0.506 against an 18.7% loss base rate). All importances are therefore near zero and the ranking is mostly noise. Read this as evidence that margin is not explained by the recorded order attributes, not as a list of true drivers.

## How to run
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook APL_Logistics_Profitability_Analysis.ipynb
# once app.py exists:
streamlit run app.py
```
Set `DATA_PATH` in the first code cell of the notebook if the CSV is in a different folder.

## Limitations
- No order date, so no time trends.
- A single currency is assumed; no cost breakdown (shipping cost is not a separate field).
- The what-if scenario does not model demand response to price.
- Cancelled and suspected-fraud exclusion is a judgement call.

## Author
Chandan Ram, Quest Infosys Foundation Group of Institutions
