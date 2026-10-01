# Bayesian Media Mix Modeling

This project implements a Bayesian Media Mix Model using Stan and Python. The analysis is performed on weekly sales and marketing spend data across ten distinct media channels, along with control variables such as seasonality and promotional events.

# Installation

## Prerequisites

- Python 3.11.9 installed and accessible as `python` in your terminal.

## Setup Virtual Environment

Create a separate virtual environment named `bmmm`:

```bash
python -m venv bmmm
```

Activate the virtual environment:

- **Windows (PowerShell):**

  ```powershell
  .\bmmm\Scripts\Activate.ps1
  ```

- **Windows (CMD):**

  ```cmd
  bmmm\Scripts\activate.bat
  ```

- **macOS/Linux:**

  ```bash
  source bmmm/bin/activate
  ```

## Install Required Packages

Upgrade core packaging tools and install all dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel --quiet
python -m pip install -r requirements.txt --quiet
```

---
 

## Dataset

- **Source**:  
  He, S. (2021) *Bayesian Media Mix Modeling using Stan*. GitHub repository.  
  Available at: [https://github.com/sibylhe/mmm_stan](https://github.com/sibylhe/mmm_stan) (Accessed: 6 April 2025)

- **Description**:  
  The dataset includes:
  - Weekly sales data
  - Marketing spend across 10 media channels
  - Control variables: seasonality indicators, promotional events etc.

#
