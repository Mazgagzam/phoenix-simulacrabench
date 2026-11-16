# External data

The submitted predictor fits on the visible rows supplied by SimulacraBench.
External survey records and parameters fitted to them are absent from the
submission. The [README](README.md) lists the pretrained checkpoints.

## Development datasets

| Source | Size and use | Access terms |
| --- | --- | --- |
| [NORC General Social Survey](https://gss.norc.org/get-the-data/stata.html) | The [2021 Stata release](https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/2021_stata.zip) supplied 4,032 respondents and 36 items for the main comparison of adaptive and fixed mixture weights. The split was 2,822 for training, 604 for development, and 606 for testing. Other experiments used the 2022 and 2024 waves (4,149 and 3,986 rows) and a cohort of 6,041 rows from 2008, 2010, and 2012. | [NORC data and citation terms](https://gss.norc.org/terms-and-conditions.html). |
| [Wellcome Global Monitor 2018](https://wellcome.org/insights/reports/wellcome-global-monitor-2018#downloads) | 149,014 source rows. The first vaccine response experiment used 9,337 respondents. | The [2018 report](https://wellcome.org/sites/default/files/wellcome-global-monitor-2018.pdf) identifies its data and text as CC BY. Separate terms for the workbook were not verified. |
| [OECD PIAAC Cycle 2](https://www.oecd.org/en/data/datasets/PIAAC-2nd-Cycle-Database.html) | Four cohorts of 6,000 rows for skills prediction experiments. A separate study used 2,028 additional training rows. | [OECD data terms](https://www.oecd.org/en/about/terms-and-conditions.html). |
| [Alrababah et al., The Dynamics of Refugee Return](https://doi.org/10.7910/DVN/FK2NEV) | 3,003 prepared rows from Lebanon and 1,286 from Jordan. Used for return intention and geographic proxy studies. | CC0 1.0, as recorded in Dataverse. |
| [psychTools BFI](https://github.com/cran/psychTools/tree/7dc591f8ac11c0ef1764cf853f3065aa60653d09) | 2,800 rows for ordinal response experiments. | The package declares GPL-2 or later. Separate terms for the dataset were not verified. |
| [Afrobarometer Round 9](https://www.afrobarometer.org/data/merged-data/) | 53,444 source rows. Descriptive checks used 4,472 training rows from Kenya, Mali, Nigeria, and Zambia. | [Afrobarometer data and citation policy](https://www.afrobarometer.org/data/data-usage-and-access-policy/). |

## Reference material

[GeoNames](https://download.geonames.org/export/dump/readme.txt) (CC BY 4.0)
and [Syrian ADM1 boundaries](https://github.com/wmgeolab/geoBoundaries/tree/9469f09592ced973a3448cf66b6100b741b64c0d/releaseData/gbOpen/SYR/ADM1)
(CC BY 3.0 IGO per the source notice) supported exploratory work on geographic
pooling. The [Iraq Camp Profiling and Intentions Survey](https://data.humdata.org/dataset/reach-iraq-camp-profiling-xv-intentions-survey-viii-2021)
and [Journey Back Home](https://doi.org/10.7910/DVN/UGI0MH) were consulted at
the questionnaire or catalog level. No respondent records from either of
those two sources were used. The organizer's synthetic sandbox was used for
software and runtime checks.
