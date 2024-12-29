# Amortized Variational Inference by Policy Search (AVIPS)

This repository contains the code and resources for my master's thesis, titled **"Amortized Variational Inference by Policy Search (AVIPS)."** The project focuses on developing and evaluating AVIPS, a new gradient-based inference algorithm that combines amortized variational inference with Gaussian Mixture of Experts and information geometric optimization.

---

## Repository Structure

### **1. `daft/`**
Contains the implementation of the non-amortized baseline algorithm, Variational Inference by Policy Search (VIPS). This serves as a reference for comparing the performance of the proposed AVIPS model.

### **2. `report/`**
Includes all the experimental results, plots, and analysis from the conducted experiments. This directory supports the evaluation and comparison of AVIPS with existing methods.

### **3. `toy_task/`**
Contains the experimental targets and proposed algorithm.

### **4. `scripts/`**
Includes configuration files and scripts used for running experiments, training models, and generating results.

### **5. `environment.yml`**
A YAML file for setting up the Python environment to reproduce the results. It contains all necessary dependencies and package versions.

---

## Highlights of the Project

1. **Algorithm Development**:
   - Implementation of AVIPS, which leverages Gaussian Mixture of Experts for efficient posterior approximation.
   - Integration of amortized variational inference with information geometric optimization for improved scalability.

2. **Baseline Comparison**:
   - The repository includes amooritzed baseline: vanilla algorithm and Sticking the Landing algorithm, and non-amortized baseline: VIPS to evaluate the advantages of AVIPS.

3. **Comprehensive Evaluation**:
   - Experimental results covering multiple toy tasks.
   - Includes metrics such as Jensen–Shannon divergence, Jeffery's divergence, ELBO, inference time.
