---

## Table of Contents
- [Overview](#overview)
- [1. Initial Exploration and Cleaning](#1-initial-exploration-and-cleaning)
- [2. Expanding the Dataset](#2-expanding-the-dataset)
- [3. The Final Dataset Emerges](#3-the-final-dataset-emerges)
- [Usage](#usage)
- [Contact](#contact)

---

## Overview

Welcome to our cinematic data odyssey! This project documents the creation of a comprehensive movie dataset curated from multiple sources to enable box office success prediction and sentiment analysis. The journey involved meticulous data exploration, cleaning, merging, and manual validation to build a dataset rich in features and ready for insightful analysis.

---

## 1. Initial Exploration and Cleaning

- **Starting Point:**  
  Began with the "Movie Industry" dataset (`movies.csv`), containing **7,669 movies**.

- **Budget Focus:**  
  Focused on ~5,300 movies with available budget information.

- **Enriching Data Completeness:**  
  Referenced "IMDb 5000 Movie Dataset" (`movie_metadata.csv`) for additional actor info.  
  Matched and merged to find **3,588 movies** common to both datasets with complete info.

---

## 2. Expanding the Dataset

- **Growing the Dataset:**  
  Explored budget data from top 400-500 movies in the "IMDb 5000+ Movies & Multiple Genres Dataset" (`IMDb 5000+.csv`) and "Top 500 Movies by Production Budget" (`Top-500-movies.csv`).

- **Cross-Reference and Deduplication:**  
  Compared "Movies Dataset" with "IMDb 5000+ Movies & Multiple Genres Dataset" to find ~2,500 unique additional movies.

- **Manual Data Verification:**  
  Filled gaps for missing budget or revenue data by web scraping and manual validation.

---

## 3. The Final Dataset Emerges

🎉 **Total Movies:** 7,119

| Feature       | Description                        |
|---------------|----------------------------------|
| Name          | Movie Title                      |
| Genre         | Primary Genre                    |
| Director      | Director’s Name                  |
| Actor 1       | Lead Actor                      |
| Actor 2       | Second Lead Actor               |
| IMDb Score    | IMDb User Rating                |
| Budget        | Movie Production Budget (USD)    |
| **Revenue**   | **Box Office Revenue (Target)** |

Our `final_dataset` is now a cinematic goldmine ready to:

- Unravel box office success patterns  
- Enable robust predictive modeling  
- Analyze genre-wise sentiment trends

---

## Usage

You can leverage this dataset to build regression models for revenue prediction, perform sentiment analysis on viewer reviews using tools like VADER, and generate compelling visualizations.

---

## Contact

If you want to collaborate or have questions, feel free to reach out!

---

> _“Data is the new film reel — each row tells a story, each column frames the drama.”_ 🎬

---

*Generated with care by Divyanshu