# Stock Market Prediction with CNNs and Hybrid CNN Transformers

Master's thesis, M.Sc. Financial Economics, Maastricht University (2026). Grade: 8.69/10.

Can a convolutional network read a price chart the way a technical analyst claims to?
This project turns 315,903 stock day observations of the Euro Stoxx 50 into chart images
and engineered indicators, trains deep learning models on them, and then tests, as
strictly as possible, whether anything is left once the usual sources of illusion are
removed.

## What the pipeline does

1. **Data.** Automated collection and cleaning of 315,903 stock day observations.
2. **Representation.** Each observation becomes a chart image plus 25 engineered
   technical indicators.
3. **Models.** Convolutional architectures and a hybrid CNN Transformer, trained on GPU
   in PyTorch.
4. **Validation.** Strict walk forward design, so no future information reaches training.
5. **Statistics.** Bootstrap confidence intervals and DeLong tests instead of a single
   accuracy number.
6. **Economics.** Quintile long short portfolios, evaluated net of transaction costs.
7. **Transfer.** Zero shot application of the trained models to cryptocurrencies.

## Why it is built this way

A model that predicts returns is easy to produce and hard to believe. Most of the work
here is spent on the part that can make the result fail: separating training and test
periods in time, testing whether a difference in performance is statistically
distinguishable from noise, and checking whether a statistical edge survives trading
costs. Several hypotheses did not survive those tests, and the thesis reports that.

## Repository layout

| Path | Contents |
| --- | --- |
| `data_pipeline_v3.py`, `data_pipeline_v4.py` | Data collection, cleaning and image generation |
| `model_convnextv2_itransformer.py` | Hybrid ConvNeXt V2 and iTransformer architecture |
| `model_efficientnetb2_transformer.py` | Hybrid EfficientNet B2 and Transformer architecture |
| `cnn_paper.py` | Baseline replication of the reference paper |
| `corrected_backtest.py` | Walk forward backtest, net of transaction costs |
| `performance_analysis.py`, `comparison_analysis.py` | Statistical tests and model comparison |
| `crypto_inference.py` | Zero shot transfer to cryptocurrencies |
| `Chapters/`, `main.tex`, `references.bib` | Thesis text |

## Stack

Python, PyTorch, pandas, NumPy, scikit-learn, LaTeX.

## Note

Research code, written for a thesis rather than for production. It is published so the
method and the results can be checked and challenged.
