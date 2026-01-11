# ALNS Loggi CVRP Solver

This project provides a minimal **Adaptive Large Neighborhood Search (ALNS)** solver for **CVRP** instances formatted in **Loggi / TSPLIB** style.

---

## Project Structure

```
project/
│
├── alns_loggi.py
└── data/
    └── instances/
        ├── Loggi-n401-k23.vrp
        ├── Loggi-n601-k19.vrp
        ├── Loggi-n701-k38.vrp
        ├── Loggi-n701-k41.vrp
        ├── Loggi-n1001-k52.vrp
        └── ... (other .vrp files)
```

All `.vrp` files must be placed in:

```
data/instances/
```

---

## Environment Setup

### Requirements

- **Python:** 3.8 – 3.11  
  > Python 3.12 is not supported due to dependency compatibility.
- **Required libraries:**
  - `numpy`
  - `numba`

---

### (Recommended) Create a Virtual Environment

From the **project root directory**:

```bash
python -m venv venv
```

Activate the environment:

#### Windows (PowerShell)

```powershell
venv\Scripts\activate
```

If script execution is blocked, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then restart PowerShell and activate again.

#### Windows (CMD)

```cmd
venv\Scripts\activate.bat
```

#### Linux / macOS

```bash
source venv/bin/activate
```

When activated, the terminal prompt will show:

```
(venv)
```

---

### Install Dependencies

```bash
pip install numpy numba
```

---

## How to Run

All commands below are executed from the **project root directory**.

---

### Run One Instance

```bash
python alns_loggi.py   --folder data/instances   --instances Loggi-n601-k42.vrp   --iters 20000   --seed 42   --time 120
```

---

### Run Multiple Instances

```bash
python alns_loggi.py   --folder data/instances   --instances Loggi-n401-k23.vrp Loggi-n601-k19.vrp   --iters 2000   --seed 17
```

---

### Run All `.vrp` Files (PowerShell)

```powershell
python alns_loggi.py `
  --folder data/instances `
  --instances (Get-ChildItem data/instances/*.vrp).Name `
  --iters 10000 `
  --seed 17
```

---

### Run All `.vrp` Files (Linux / macOS)

```bash
python alns_loggi.py   --folder data/instances   --instances *.vrp   --iters 10000   --seed 17
```

---

## Output

For each instance, the solver prints:

```
Loading: data/instances/Loggi-n401-k23.vrp
  n=401 cap=100 depot=1
  distance matrix loaded ((401, 401))
✔ Done Loggi-n401-k23.vrp | best=145016.00 | time=0.90s
```

---

## Notes

- Ensure Python version is within **3.8 – 3.11**.
- Always activate the virtual environment before running the solver.
- File paths are resolved relative to the project root.
