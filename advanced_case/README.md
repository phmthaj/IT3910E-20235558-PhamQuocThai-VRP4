# ALNS-Based MDVRP Solver with Trip Limits

This project implements an **Adaptive Large Neighborhood Search (ALNS)** solver for the
**Multi-Depot Vehicle Routing Problem (MDVRP)** with practical business constraints.

The system is designed for **large-scale instances** and supports **parallel execution** with **Numba JIT acceleration**.

---

## Project Structure

```text
advanced_case/
│
├── dataset/                # Input VRP datasets (.vrp)
│   └── VRP4_20k_Dataset.vrp
│
├── Log/                    # Output solutions & execution logs
│   ├── *.sol
│
├── opt_code.py              # Main ALNS solver (entry point)
├── DataGenerate.ipynb       # Dataset generation (optional)
├── Visualize.ipynb          # Result visualization (optional)
├── requirement.txt          # Python dependencies
└── README.md                # Project documentation
```

---

## Key Feature: Trip Limit (Quota Constraint)

The solver enforces a **Trip Limit (Quota)** for **each vehicle type at each depot**.

### Cost Rule

* The first **N trips** are charged at the **base cost (1×)**.
* Any additional trips beyond **N** are penalized at **3× the base cost**.

> This mechanism discourages excessive vehicle usage and models real-world operational constraints.

---
1. Requirements

Python: 3.8 – 3.11

Python 3.12 is not recommended (Numba compatibility issues).

Libraries:

numpy

numba

2. Create a Virtual Environment (Recommended)

From the project root directory:

python -m venv venv


Activate the environment:

Windows (PowerShell)
venv\Scripts\activate


If script execution is blocked, run once:

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned


Then restart PowerShell and activate again.

Windows (CMD)
venv\Scripts\activate.bat

Linux / macOS
source venv/bin/activate


When activated, the terminal prompt will show:

(venv)

3. Install Dependencies

All commands below must be executed from the advanced_case directory.

cd advanced_case


Install required libraries:

pip install -r requirement.txt


Or install manually:

pip install numpy numba

---

## 3. Usage

All experiments must be executed **from the `advanced_case` directory**.

### Basic Run (Default Trip Limit = 30)

```bash
python opt_code.py --instance dataset/VRP4_20k_Dataset.vrp
```

---

### Run with Custom Trip Limit

Set the quota to **10 trips per vehicle type per depot**.
Trips beyond the quota are penalized at **3× cost**.

```bash
python opt_code.py \
  --instance dataset/VRP4_20k_Dataset.vrp \
  --limit_trips 10
```

---

### Run with Advanced Tuning

Run ALNS for **1000 iterations** with a **time limit of 1000 seconds**:

```bash
python opt_code.py \
  --instance dataset/VRP4_20k_Dataset.vrp \
  --iters 1000 \
  --time 1000 \
  --limit_trips 9
```

---

## 4. Command-Line Arguments

| Argument        | Required | Default | Description                                 |
| --------------- | -------- | ------- | ------------------------------------------- |
| `--instance`    | Yes      | —       | Path to input `.vrp` file (from `dataset/`) |
| `--limit_trips` | No       | 30      | Max trips before penalty (3× cost)          |
| `--iters`       | No       | 300     | Number of ALNS iterations                   |
| `--time`        | No       | 1000    | Time limit (seconds)                        |
| `--seed`        | No       | 42      | Random seed for reproducibility             |

---

## 5. Output

After execution, the solver automatically writes results to the **`Log/` directory**.

### Generated Files

* **Solution file (`.sol`)**

### Solution File Contains

#### Header Information

* Total Cost
* Execution Time
* Number of Vehicles Used
* Number of Penalized Trips

#### Route Details

* Routes grouped by **Depot**
* Penalized trips are clearly marked as:

```text
[PENALIZED x3]
```

---

## Notes

* The solver is optimized using **Numba JIT** for fast distance computation.
* Designed to scale to **thousands of nodes**.
* Supports **multi-depot**, **multi-commodity**, and **quota-based cost policies**.
* CPU usage may reach **100%** during execution due to parallel processing — this is expected behavior.

---

## License

This project is intended for **academic and research purposes**.
