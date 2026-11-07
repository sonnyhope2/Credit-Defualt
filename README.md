# Does gradient boosting beat logistic regression out of time?


**Mortgage default prediction on Freddie Mac loan-level data, 2000–2024 vintages**

Trained on loans originated in 2000–2012 and tested on loans originated in 2015–2017, LightGBM barely beats logistic regression: out-of-time AUC 0.801 against 0.799, a gain whose 95% bootstrap interval (-0.004 to +0.008) includes zero. A monotone-constrained LightGBM does slightly better (AUC 0.805, gain +0.007, interval +0.0004 to +0.013), and its edge widens on the 2022–2024 vintages. But every model loses about 0.06 AUC between an in-time and an out-of-time test, and every model over-predicts defaults about twofold. Out of time, the change in economic conditions matters far more than the choice of algorithm.

## Question

Gradient boosting usually beats logistic regression on a random test split. But credit models are built on past loans and used on future ones, and the economy changes in between. This project asks:

1. Does gradient boosting (LightGBM) still beat logistic regression when tested **out of time**, on loans originated after the training period?
2. Is any gain large enough to justify losing a model whose every coefficient can be read off directly?

## Data

[Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), sample files: a random sample of 50,000 loans per origination year (vintage), 2000–2024, with monthly performance records up to the release's performance cutoff (printed by `make_dataset.py`). The Standard Dataset covers fully amortising, fixed-rate, full-documentation mortgages acquired by Freddie Mac.

The data is free to download after registering, but may not be redistributed, so it is not in this repository (see [Reproduce](#reproduce)).

## Method

### Defining default

A loan **defaults** if, within **24 months of its first payment**, it becomes 90+ days delinquent, is acquired as real-estate owned (REO), or ends in a credit-related termination (third-party sale, short sale or charge-off, REO disposition).

- **Months on book come from the dates, not the dataset's loan-age field**, because loan age restarts after a modification.
- **The window matters.** For the 2005 vintage, 1.05% of loans default within 24 months, 2.09% within 36 and 6.36% within 60, against 11.04% at any point. Crisis-era defaults arrived late, so a short window understates pre-crisis risk. I use 24 months so that more recent vintages are fully observed; `WINDOW` in `src/config.py` can be set to 36 as a robustness check.
- **Censoring:** a loan is only used if its whole window ends before the data does. The rule depends only on the first payment date, never on what happened to the loan. Loans that prepay inside the window count as non-defaults, since they can no longer default.

### Out-of-time design

| Split | Vintages | Used for |
|---|---|---|
| Train | 2000–2012 (80%) | fitting |
| In-time test | 2000–2012 (random 20%) | the "usual" random-split benchmark |
| Validation | 2013–2014 | choosing settings, early stopping, recalibration |
| **Out-of-time test** | **2015–2017** | **the main result; untouched until evaluation** |
| Monitoring | 2018–2024 | stability over time |

The test period deliberately ends with the 2017 vintage. From 2018 onwards, 24-month windows overlap the COVID-19 forbearance period, when the delinquency status (based on the last paid instalment) can show loans in forbearance as 90+ days late. Those vintages are reported separately.

### Features

Only information known at origination, so nothing leaks the outcome:

- **Numeric:** credit score, DTI, LTV, mortgage insurance %, log loan amount, and **rate spread**, the loan's rate minus the median rate of loans from the same quarter and term. Raw interest rates mostly track the market (about 6% in 2006, 4% in 2015), so they would teach the model which year a loan came from. The spread keeps the part of pricing that reflects the borrower's risk.
- **Binary:** second lien (CLTV above LTV), more than one borrower (coded consistently despite a 2018 definition change), multi-unit property, HARP refinance, super-conforming.
- **Categorical:** occupancy, loan purpose, channel, property type, first-time buyer, state, term group.

"Not available" placeholder codes (e.g. a credit score of 9999) are converted to missing values. Fields populated only for recent years (VantageScore, property valuation method) are excluded, as are seller and servicer names and postal codes.

### Models

| Model | Interpretability |
|---|---|
| Logistic regression | One coefficient per feature (numeric features standardised, median-imputed with a missing flag) |
| Logistic regression, binned | Numeric features cut into training-set deciles, with missing values as their own bin, as in a credit scorecard: non-linear but still additive |
| LightGBM | Black box, explained with SHAP |
| LightGBM, monotone | Same settings, but risk is forced to rise with LTV, DTI and rate spread and to fall with credit score |

The binned logistic regression gives the interpretable side a fair chance by letting it capture non-linear effects. The monotone LightGBM is a middle ground that a model validator would find easier to sign off.

No resampling or class weights are used. Rebalancing would distort the predicted probabilities, and probabilities are what a lender prices and provisions with. Regularisation strength (logistic regression) and tree settings (LightGBM, with early stopping) are chosen on validation AUC.

### Evaluation

- **Discrimination:** AUC and Gini, precision-recall AUC (more informative than AUC when defaults are rare), KS, and the share of defaults captured in the riskiest 10% of loans.
- **Uncertainty:** paired bootstrap (500 resamples) of the AUC gain over logistic regression.
- **Calibration:** observed/expected default rate, calibration slope, Brier score and reliability curves, before and after Platt recalibration on the validation vintages.
- **Stability:** AUC by vintage, and the population stability index (PSI) between training and test vintages.

## Results

The out-of-time test set is 150,000 loans from the 2015–2017 vintages, of which 980 (0.65%) defaulted within 24 months. Full tables are in `reports/results.md`.

### Discrimination

| Model | AUC | PR-AUC | KS | Defaults in riskiest 10% |
|---|---|---|---|---|
| Logistic regression | 0.799 | 0.0289 | 0.461 | **43.3%** |
| Logistic regression, binned | 0.796 | **0.0292** | 0.454 | 41.9% |
| LightGBM | 0.801 | 0.0273 | 0.462 | 40.5% |
| LightGBM, monotone | **0.805** | 0.0284 | **0.475** | 42.2% |

The four models are within 0.01 AUC of each other, and which one "wins" depends on the metric. LightGBM has the second-highest AUC but the lowest precision-recall AUC. It also finds the fewest defaults in the riskiest 10% of loans, which is where plain logistic regression does best.

### Is the gain real?

| Model | AUC gain over logistic regression | 95% interval | Resamples with a gain |
|---|---|---|---|
| Logistic regression, binned | -0.0032 | -0.0072 to +0.0009 | 5.6% |
| LightGBM | +0.0020 | -0.0037 to +0.0077 | 74.6% |
| LightGBM, monotone | +0.0066 | +0.0004 to +0.0133 | 98.4% |

Only the monotone LightGBM's interval excludes zero, and only just. Its bootstrap two-sided p-value is about 0.03, which would not survive a Bonferroni correction for three comparisons (about 0.10). The interval also understates uncertainty: it resamples test loans, but not the training process, and loans from the same vintage share economic conditions.

### In time vs out of time

| Model | In-time AUC | Out-of-time AUC | Change |
|---|---|---|---|
| Logistic regression | 0.858 | 0.799 | -0.059 |
| Logistic regression, binned | 0.859 | 0.796 | -0.063 |
| LightGBM | 0.863 | 0.801 | -0.062 |
| LightGBM, monotone | 0.857 | 0.805 | **-0.051** |

Every model loses about 0.06 AUC, roughly ten times the gap between the models. Unconstrained LightGBM leads in time (+0.005 over logistic regression), but most of that edge disappears out of time (+0.002). That suggests its extra flexibility partly fitted patterns specific to 2000–2012. The monotone version loses the least: its constraints act as regularisation that carries over across time.

### Calibration

| Model | Mean predicted PD | Observed / expected | Calibration slope | After recalibration: observed / expected | After recalibration: slope |
|---|---|---|---|---|---|
| Logistic regression | 1.32% | 0.50 | 0.99 | 1.28 | 1.08 |
| Logistic regression, binned | 1.37% | 0.48 | 0.83 | 1.24 | 0.92 |
| LightGBM | 1.42% | 0.46 | 0.89 | 1.15 | 0.84 |
| LightGBM, monotone | 1.43% | 0.46 | 0.99 | 1.15 | 0.83 |

Observed default rate: 0.65%. Brier scores are 0.0066–0.0067 as trained and 0.0064–0.0065 after recalibration.

As trained, all four models predict about twice the observed default rate. The training vintages include the 2005–2008 crisis loans, and the test vintages default far less. The PSI results below show this is not because the test borrowers look safer: the model score distributions barely moved (PSI 0.03–0.04). The difference is the economic environment, which a model built only on origination data cannot see.

Logistic regression and the monotone LightGBM have calibration slopes of 0.99. Their probabilities are wrong in level but right in spread. The unconstrained LightGBM (0.89) and the binned regression (0.83) are too extreme.

Recalibrating on 2013–2014 removes most of the level error but overshoots. Default rates rose from 0.48% in the validation vintages to 0.65% in the test vintages, so the recalibrated models under-predict by 15–28%. Calibration holds for a period, not for ever, which is why PD models are monitored and recalibrated in practice.

### Stability by vintage

| Vintage | Default rate | Logistic regression | Binned | LightGBM | LightGBM, monotone |
|---|---|---|---|---|---|
| 2013 (valid) | 0.47% | 0.833 | 0.834 | 0.845 | 0.847 |
| 2014 (valid) | 0.49% | 0.782 | 0.786 | 0.804 | 0.805 |
| 2015 (test) | 0.51% | 0.793 | 0.791 | 0.802 | 0.806 |
| 2016 (test) | 0.66% | 0.802 | 0.798 | 0.798 | 0.805 |
| 2017 (test) | 0.79% | 0.795 | 0.791 | 0.798 | 0.800 |
| 2018* | 2.56% | 0.729 | 0.726 | 0.728 | 0.733 |
| 2019* | 4.58% | 0.709 | 0.709 | 0.714 | 0.715 |
| 2020* | 1.62% | 0.728 | 0.724 | 0.734 | 0.735 |
| 2021 | 0.93% | 0.774 | 0.774 | 0.775 | 0.780 |
| 2022 | 1.78% | 0.738 | 0.751 | 0.757 | 0.775 |
| 2023 | 1.75% | 0.759 | 0.773 | 0.772 | 0.788 |
| 2024 | 1.71% | 0.777 | 0.791 | 0.793 | 0.815 |

\* The 24-month window overlaps COVID-19 forbearance.

![Stability by vintage](reports/figures/vintage_stability.png)

The 2018–2019 vintages show why the test stops at 2017. Their default rates jump to 2.6% and 4.6% as forbearance shows up as 90+ day delinquency, and every model's AUC falls to about 0.71–0.73: a pandemic shock is hard to predict from origination data.

For the 2022–2024 vintages, the monotone LightGBM pulls clearly ahead of linear logistic regression, by 0.03–0.04 AUC, and the binned regression also gains on the linear one. One possible reason is that later borrowers sit in ranges where a straight-line effect extrapolates badly; DTI had already shifted moderately by 2015–2017 (PSI 0.18). This is not tested here. The 2024 row covers only the 20,949 loans whose 24-month window is complete.

### Population stability (training vintages vs 2015–2017)

| Variable | PSI | Reading |
|---|---|---|
| Credit score | 0.073 | stable |
| DTI | 0.178 | moderate shift |
| LTV | 0.080 | stable |
| MI % | 0.095 | stable |
| Log loan amount | 0.248 | moderate shift |
| Rate spread | 0.043 | stable |
| Model scores | 0.030–0.042 | stable |

Loan size shifts the most, largely because loan amounts are in nominal dollars and rise with house prices. A size measure relative to the conforming loan limit would be more stable. The rate spread, built to remove market-wide rate movements, is the most stable feature.

![ROC and precision-recall curves](reports/figures/roc_pr_oot.png)
![Calibration](reports/figures/calibration_oot.png)

## Interpretability

SHAP for LightGBM and the logistic regression's coefficients tell the same main story:

| Rank | LightGBM (mean \|SHAP\|) | Logistic regression (largest coefficients) |
|---|---|---|
| 1 | Credit score (0.52) | More than one borrower (odds ratio 0.42) |
| 2 | More than one borrower (0.30) | Credit score, per standard deviation (0.52) |
| 3 | LTV (0.25) | Loan term (15-year vs 30-year) |
| 4 | Loan term (0.17) | State (Florida, Louisiana, Mississippi, Nevada highest) |
| 5 | DTI (0.17) | Super-conforming (0.68) |

- **Credit score** is the strongest driver in both models.
- **Number of borrowers:** single-borrower loans have about 2.4 times the default odds of multi-borrower loans, all else equal. Two incomes make three missed payments in a row far less likely.
- **Loan term:** 15-year loans are much safer than 30-year ones (a gap of about 0.87 in log-odds). Borrowers who can afford higher payments tend to be stronger, and they build equity faster.
- **State:** Florida and Nevada were at the centre of the housing crisis. Louisiana and Mississippi may reflect Hurricane Katrina (August 2005) pushing 2004–2005 loans into delinquency, which would be a disaster effect rather than credit quality. This hypothesis is not tested here.
- **Channel:** broker loans are riskier and retail loans safer, consistent with the pre-crisis literature.
- **Constraints shrink reliance:** the monotone LightGBM gives every feature less weight; state importance halves, from 0.122 to 0.061.
- **LTV** ranks third in LightGBM but is not among logistic regression's 15 largest terms. LTV and mortgage insurance are closely related (insurance is generally required above 80% LTV), so the linear model may split one effect between the two; its individual LTV coefficient should not be over-read.

> **TODO:** describe where the partial dependence curves bend and where the models disagree.

![Partial dependence](reports/figures/partial_dependence.png)
![SHAP importance](reports/figures/shap_importance.png)
![SHAP beeswarm](reports/figures/shap_beeswarm.png)
![Local explanation](reports/figures/local_explanation.png)

## Is it worth it?

**For unconstrained gradient boosting, no.** Out of time it adds 0.002 AUC, statistically indistinguishable from zero. It captures fewer defaults in the riskiest decile and gives less well-shaped probabilities than logistic regression, and it needs SHAP to explain every decision.

**The monotone LightGBM is a closer call.** Its gain is small (+0.007 AUC), borderline significant, and grows on the most recent vintages. Its constraints guarantee the directions a credit committee expects, and an automatic check confirms them. That makes it a defensible challenger model, but not a clear replacement for a logistic regression that is nearly as accurate and can be read from a table.

**The larger lesson** is that every model lost about 0.06 AUC and over-predicted defaults twofold when the economy changed. Out of time, effort spent on macroeconomic covariates, monitoring and recalibration is likely to pay off more than effort spent on a more flexible algorithm.

## Limitations

- **Sample and scope:** 50,000 loans per vintage from one agency's fixed-rate, full-documentation book. Results may not carry over to subprime, adjustable-rate or non-agency loans.
- **No macroeconomic variables:** the models see the borrower and the loan but not house prices or unemployment. They rank risk within a vintage but cannot anticipate a crisis, which shows up as miscalibration across vintages.
- **Bootstrap intervals are optimistic:** they ignore training randomness and correlation within vintages, and three comparisons were made.
- **Fixed-window label:** prepayment is a competing risk, treated here as non-default. A survival model would use the timing information directly.
- **COVID-era labels:** forbearance inflates 90+ day delinquency for 2018–2020 vintages.
- **Rate spread proxy:** the market rate is the median of sampled loans in the same quarter, not a published series such as Freddie Mac's PMMS.
- **Seasoned loans:** performance starts when Freddie Mac acquires a loan, so any delinquency before acquisition is not seen.
- **Small hyperparameter search**, chosen for speed.
- **SHAP explains the model, not causation.**

## Possible extensions

House price and unemployment covariates; a discrete-time hazard model with prepayment as a competing risk; loan size relative to the conforming limit; testing the Katrina hypothesis by state and vintage; the full dataset instead of the sample; a weight-of-evidence scorecard; a 36-month window as a robustness check.

## Reproduce

1. Register at Freddie Mac Clarity and download the **sample** files (`sample_YYYY.zip`) for 2000–2024 from the Single-Family Loan-Level Dataset page.
2. Unzip each into `data/`, giving `data/sample_2005/sample_orig_2005.txt` and so on. If the data lives elsewhere, set the `SFLLD_RAW_DIR` environment variable to that folder, or edit `RAW_DIR` in `src/config.py`.
3. Run:

```bash
python -m pip install -r requirements.txt
python src/make_dataset.py   # one row per loan
python src/train.py          # fits and selects the four models
python src/evaluate.py       # tables in reports/results.md, figures in reports/figures/
python src/explain.py        # SHAP, partial dependence, coefficients
```

## Project structure

```
src/
  config.py        paths, label window, vintage split
  load_data.py     reads raw files, recodes placeholders, finds first default month
  make_dataset.py  step 1: builds data/processed/loans.parquet
  features.py      feature engineering, label, censoring, split
  modelling.py     the four model pipelines
  train.py         step 2: model selection on validation vintages
  evaluate.py      step 3: discrimination, calibration, bootstrap, stability
  explain.py       step 4: SHAP, partial dependence, odds ratios
reports/           results tables and figures (committed)
data/, models/     generated locally, not committed
```
