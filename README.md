# Transferring Knowledge to Tackle Machine Learning Bias when Sensitive Information is not Available

This repository contains supplementary materials related to the paper titled "Transferring Knowledge to Tackle Machine Learning Bias when Sensitive Information is not Available" submitted to the Knowledge Capture 2023 conference. The paper addresses the challenge of mitigating bias in machine learning algorithms when access to sensitive attributes is restricted due to privacy or legal concerns. The proposed approach leverages covariance analysis and alternative indicators to indirectly address bias in scenarios where sensitive attribute data is unavailable.

## Paper Abstract

Bias in Machine Learning (ML) algorithms continues to draw much attention. Many existing methods to mitigate bias in ML require access to sensitive attributes (e.g. gender) to train ML to not bias against groups represented by such attributes (e.g. females). However, this can be a challenge when such data is unavailable due to privacy or legal restrictions, knowing that ML algorithms could still indirectly bias against such attributes. This research proposes a new approach using covariance analysis to address bias in ML models when sensitive attributes are absent. The method involves identifying alternative indicators (such as maternity leave) that indirectly relate to sensitive attributes in more permissive data environments (e.g., countries with open data policies). This knowledge is then transferred to stricter data settings (e.g., countries with strong data privacy regulations) to reduce bias in ML models trained under this limited data availability. Evaluation on US Census data demonstrates the effectiveness of this approach in identifying relevant indicators within different US states and mitigating bias in ML models trained in scenarios lacking sensitive attributes.

## Contents

This repository includes the following resources:

- **Paper (Future Update)**: Once the paper is reviewed, we plan to upload the camera ready paper titled "Transferring Knowledge to Tackle Machine Learning Bias when Sensitive Information is not Available" that provides an exploration of the proposed approach, methodology, experiments, and results.

- **Results**: The supplementary results discussed in the paper are included for reference.

- **Code (Future Update)**: Once the paper is reviewed, we plan to release the code used for implementing the bias mitigation approach. The code will be provided along with instructions to replicate the experiments and apply the proposed method.
