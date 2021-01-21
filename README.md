# Software-related Chat Disentanglement
This repository contains dataset and DLD metrics analysis, and the research files of empirical study and investigation.

Our project is public at [https://github.com/disensoftware/disentanglement-for-software](https://github.com/disensoftware/disentanglement-for-software)
## SOTA models
The state-of-art models and datasets are available as follow:

- [BiLSTM](https://github.com/layneins/e2e-dialo-disentanglement): Author: Meheri et al., implemented by Liu et al.
- [FF](https://github.com/jkkummerfeld/irc-disentanglement/zipball/master): Author: Kummerfeld et al., implemented by Kummerfeld et al.
- [BERT](https://github.com/layneins/e2e-dialo-disentanglement): Author: Li et al., implemented by Liu et al.
- [E2E](https://github.com/layneins/e2e-dialo-disentanglement): Author: Liu et al., implemented by Liu et al.
- [PtrNet](https://github.com/vode/onlinePtrNet_disentanglement): Author: Yu et al., implemented by Yu et al.

## Datasets
Our dataset is constructed with two different formats: [original](data/proposed_dataset/original_format) and [json](data/proposed_dataset/json_format). Various data formats are intended for different model inputs. 

Moreover, we introduce the addresses of traditional symbolic datasets for referencing and analyzing.

## Metrics
The source code of traditional metric is available in [utils.py](v1_code/utils.py), including NMI, ARI and Shen-F.

The F1 and DLD require the reformation of predicted data. Since the preprocessing data of intermediate steps are private for all works,
we only provide the 1st version of reformat code [reformat_e2d_source_data.py](v1_code/reformat_data.py) for reference without bug fixing.

If you need the complete data of annotation and preprocessing, please contact us via [ziyou2019@iscas.ac.cn](mailto:ziyou2019@iscas.ac.cn)

## Experiment Data
We provide the experiment data of each models and metrics in [experiment](./experiment) fold.

- Data of models comparison experiment: [model_comparison](./experiment/model_comparison).
- Data of metrics comparison experiment: [metric_comparison](./experiment/metric_comparison).

The performance of DLD is shown as follows:

| Analysis | RMSE | MAE | PEA | IST | PST | ANOVA
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| NMI | 0.38 | 0.34 | 0.08 | e-55 | e-46 | e-55 |
| ARI | 0.37 | 0.32 | 0.02 | e-19 | e-17 | e-19 |
| Shen-F | 0.41 | 0.36 | 0.17 | e-69 | e-59 | e-69 |
| F1 | 0.19 | 0.14 | 0.85 | e-4 | e-14 | e-4 |
| DLD | 0.08 | 0.07 | 0.92 | 0.51 | 0.31 | 0.51 |


## Diagrams
All diagrams of article are available in [diagrams](./diagrams) directory.

- [Performances](./diagrams/performances): The original and retraining model performances.
- [Metrics comparison](./diagrams/metrics_comparison): The prove of DLD superiority.
- [Bad Cases](./diagrams/bad_cases): The card sorted results of incorrect disentanglement.
