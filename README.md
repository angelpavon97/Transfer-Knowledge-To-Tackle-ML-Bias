# Transferring Knowledge to Tackle Machine Learning Bias when Sensitive Information is not Available

This repository contains supplementary materials related to the paper titled "Transferring Knowledge to Tackle Machine Learning Bias when Sensitive Information is not Available" (under review). The paper addresses the challenge of mitigating bias in machine learning algorithms when access to sensitive attributes is restricted due to privacy or legal concerns. The proposed approach leverages covariance analysis and alternative indicators to indirectly address bias in scenarios where sensitive attribute data is unavailable.

## Paper Abstract

Bias in machine learning (ML) algorithms remains a significant concern. Many existing methods for mitigating bias in ML rely on access to sensitive attributes (e.g., gender, ethnicity, disability) to identify and address biases towards groups represented by these attributes (e.g., females, minority ethnic communities, users with disabilities, etc.). However, the unavailability of this information poses challenges in creating fair algorithms, as measuring their potential biases becomes more difficult. This study introduces a novel approach using covariance-based analysis to enhance the fairness of ML models generated with data where sensitive attributes are absent. The methodology involves identifying proxies for sensitive attributes (e.g., part-time work being a proxy for gender) in data environments where such attributes are available, and applying this knowledge to reduce the bias of ML models trained under limited data availability. The novelty of the approach lies in the automation of this process, where domain experts are not required to identify such proxy attributes. Evaluation conducted on US Census data demonstrates the effectiveness of this approach in enhancing the fairness of ML models trained under conditions lacking sensitive demographic data.

## Contents

This repository includes the following resources:

- **Paper (Future Update)**: Once the paper is reviewed, we plan to upload the camera-ready paper titled "Transferring Knowledge to Tackle Machine Learning Bias when Sensitive Information is not Available" which provides an exploration of the proposed approach, methodology, experiments, and results.

- **Results**: The supplementary results discussed in the paper are included for reference.

- **Code**: Once the paper is reviewed, we plan to make public the code used for our experiments.

## Repository structure

- **results**: Folder containing the results of the experiments, with extra images to facilitate the analysis of the results.
- ***apply_metrics.py***: Python file that contains the functions to compute the fairness metrics.
- ***machine_learning.py***: Python file that contains the functions to apply machine learning algorithms.
- ***preprocessing.py***: Python file that contains the functions to process the data.
- ***requirements.txt***: File containing the requirements to execute our experiments.
- ***statistical_measures.py***: Python file that contains the functions to compute the statistical tests needed for some bias mitigation approaches.
- ***preprocessing.py***: Python file that contains the functions to visualize the data.
- ***WS_FolkTables.ipynb***: Jupyter Notebook to execute all the experiments.
- ***WS_AnalyseResults.ipynb***: Jupyter Notebook to analyse all the experiments.
