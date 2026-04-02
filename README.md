# Multi-Objective Resource-Aware AutoML for Lightweight Model Generation on Tabular Data

This repository contains the implementation and experiments for the undergraduate thesis:

**"Multi-Objective Resource-Aware AutoML for Lightweight Model Generation on Tabular Data"**

Prepared and submitted by:

* Dorothy C. Salva
* Jeff Lawrence C. Balbuena
* Vex Ivan Sumang

In partial fulfillment of the requirements for the degree of **Bachelor of Science in Computer Science**.

---

## Overview

This project proposes a hybrid optimization framework for Automated Machine Learning (AutoML), designed to generate lightweight and efficient models for tabular data.

The approach integrates:

* **Non-dominated Sorting Genetic Algorithm II (NSGA-II)**
* **Meta-learning mechanisms**

to intelligently explore and optimize machine learning configurations.

---

## Objectives

The system evaluates candidate AutoML configurations based on multiple competing objectives:

* **Predictive Performance** – Accuracy and model effectiveness
* **Computational Efficiency** – Resource usage and runtime performance

This enables the discovery of optimal trade-offs using multi-objective optimization.

---

## Methodology

The proposed framework:

1. Uses NSGA-II to evolve candidate solutions
2. Applies meta-learning to guide the search process
3. Produces a Pareto front of optimal model configurations

---

## Key Features

* Multi-objective optimization
* Lightweight model generation
* Resource-aware AutoML
* Pareto front analysis
