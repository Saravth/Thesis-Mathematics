This is the code used in the thesis: 



**Learning to simulate tail risk scenarios: A comparative analysis of WGAN-GP, Tail-GAN, and Expectile-GAN.**



This repository includes the implementation of WGAN-GP (Gulrajani et al. 2017), Tail-GAN (Cont et al., 2025) and Expectile-GAN.





##### **Repository Structure \& File Descriptions**



###### **1. Core Model Implementations**

'wgan\_gp.py' Standard implementation of WGAN-GP used as our baseline for full-distribution matching.

'tail\_gan.py' Implementation of Tail-GAN using the Fissler-Ziegel scoring function to simultaneously track and generate $\\text{VaR}\_\\alpha$ and $\\text{ES}\_\\alpha.

'expectile\_gan.py' Implementation of our Expectile-GAN framework. Includes the Generator, ExpectileDiscriminator, and an asymmetric loss scorer.





###### **2. Theory Verification Scripts**

'scorefunction.py' Maps out the Fissler-Ziegel scoring function landscape as a 3D surface grid. This script replicates Figure 1 from the original Tail-GAN paper 'Tail-GAN: Learning to Simulate Tail Risk Scenarios (Cont et al., 2025).

'expectile\_table.py' Recreates the exact theoretical quantile-to-expectile map for a standard normal distribution ($Z \\sim \\mathcal{N}(0,1)$) using Brent's root-finding method and a contraction mapping loop.





###### **3. Empirical Visualizations \& Experiment Runners**

'run\_comparison.py' Script to train and compare WGAN-GP and Tail-GAN models side-by-side using synthetic Gaussian data.

'run\_sp500.py' Script that runs WGAN-GP and Tail-GAN models on real S\&P 500 return data.

'expectile\_gan\_visuals.py' Trains Expectile-GAN at a single risk level ($\\tau = 0.05$) on both Gaussian and S\&P 500 data. It outputs distribution plots and LaTeX tables.

'expectile\_gan\_taus\_visuals.py' Advanced version that trains Expectile-GAN across multiple risk levels ($\\tau = 0.01, 0.05, 0.10$) at the same time.

'gaussian\_comparison.py' Comprehensive synthetic pipeline. Runs a 5-way test across all models (WGAN-GP, single-alpha Tail-GAN, multi-alpha Tail-GAN, single-tau Expectile-GAN, and multi-tau Expectile-GAN) using pure Gaussian data and outputs 9 detailed analysis plots.

'sp500\_comparison.py' Comprehensive empirical pipeline. Downloads real S\&P 500 history (2016–2023), runs all models (WGAN-GP, single-alpha Tail-GAN, multi-alpha Tail-GAN, single-tau Expectile-GAN, and multi-tau Expectile-GAN) using rolling 50-day windows, and outputs 9 detailed analysis plots.









