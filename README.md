# 🚀 Spaceship Titanic: Predict Passenger Destinations

Welcome aboard the **Spaceship Titanic** — a futuristic machine learning challenge set in the year **2912**. Your mission: to determine which passengers were mysteriously **transported to an alternate dimension** after the ship encountered a spacetime anomaly.

This repository contains the full workflow to solve the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic), including data preprocessing, feature engineering, modeling, and evaluation.

---

## 📘 Project Description

The *Spaceship Titanic* was a massive interstellar passenger liner traveling from Earth to three newly colonized exoplanets. But en route to **55 Cancri E**, it collided with a **spacetime anomaly**, causing around half of its passengers to vanish into another dimension.

Recovered records from the ship's computer system now offer clues. You are challenged to build a predictive model that classifies whether a passenger was **Transported** (`True`) or **not** (`False`), based on available passenger information.

---

## 🧠 Objectives

- 🧼 Clean and preprocess complex and partially missing data
- 🧪 Engineer meaningful features from raw categorical and numerical inputs
- 🤖 Train multiple machine learning models including XGBoost, LightGBM, Random Forests, and Logistic Regression
- 📊 Evaluate model performance using **accuracy** as the primary metric
- 🚀 Prepare submission for Kaggle leaderboard

---

## 📂 Dataset Overview

### Input Features:
| Feature        | Description |
|----------------|-------------|
| `PassengerId`  | Unique ID (e.g., '0001_01') indicating group and passenger number |
| `HomePlanet`   | Origin planet of the passenger |
| `CryoSleep`    | Whether the passenger was in suspended animation |
| `Cabin`        | Cabin number in the format deck/num/side (e.g., 'C/123/P') |
| `Destination`  | Final destination (e.g., TRAPPIST-1e, 55 Cancri e) |
| `Age`          | Age of the passenger |
| `VIP`          | Whether the passenger paid for luxury service |
| `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` | Amount spent in each onboard service |
| `Name`         | Passenger’s name (not used in prediction) |

### Target:
- `Transported` — **True** if the passenger was transported to an alternate dimension, otherwise **False**

---

## 🛠️ Setup Instructions

### Installation
Clone the repository and install the required libraries:

```bash
git clone https://github.com/yourusername/spaceship-titanic.git
cd spaceship-titanic
pip install -r requirements.txt
