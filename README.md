Q-Learning Based Financial Trading Strategy with Hyperparameter Optimization

A reinforcement learning framework to optimize a trading strategy using **Q-Learning** with discrete state-action representation, applied to **NIFTYBEES ETF** data. Hyperparameters like learning rate (`alpha`) and discount factor (`gamma`) are optimized using **Optuna**, aiming to **maximize final portfolio returns**.

Designed with simplicity, speed, and practical impact.


Objective

This project explores whether a **Q-learning agent** can learn profitable trading behavior using minimal signals (like 5-day moving averages) in Indian equity markets. The approach includes:

- Constructing a **discrete environment** with hand-crafted state representations
- Training with **reward as portfolio value**
- Tuning learning parameters (`α`, `γ`) using **Optuna**
- Testing on out-of-sample financial data



Key Features

| Component                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Algorithm**            | Tabular Q-Learning (Buy / Sell / Hold)                                     |
| **State Space**          | `4` states from binary momentum (`u`) and position (`t`)                   |
| **Reward**               | Total portfolio value (absolute INR)                                       |
| **Data**                 | `NIFTYBEES.NS` via `yfinance` (2016–2025 split)                             |
| **Tuning Framework**     | `Optuna` used to optimize `alpha` and `gamma`                               |
| **Performance Speed-up** | Full vectorization using `NumPy` + progress bar using `tqdm`               |

---

##State & Action Design

- **State (`s`)**: Tuple of momentum signal (`u`) × position (`t`), i.e., 4 states total  
  - `u = 1` if `Close > MA5`, else `0`
  - `t = 1` if holding stock, else `0`
- **Actions (`a`)**:
  - `0 = Buy`, `1 = Sell`, `2 = Hold`

---

##Hyperparameter Search

We define:

```python
alpha ∈ [0.01, 0.99]      # Learning rate
gamma ∈ [0.01, 0.99]      # Discount factor
