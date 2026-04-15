# environment scripts in the scripts directory

## Windows (PowerShell)

### setup_env.ps1
Creates the virtual environment and installs dependencies.

### activate_env.ps1
Activates the environment.

### run_coverage.ps1
Runs the tests in the tests directory and outputs the coverage report in the test directory.
You have to add new tests to this script.

**To run the tests (from the repo root):**

```
.\scripts\setup_env.ps1
.\scripts\activate_env.ps1
.\scripts\run_coverage.ps1
```

## macOS / Linux (bash)

### setup_env.sh
Same as `setup_env.ps1`: creates `venv` and installs from `requirements.txt`.

### activate_env.sh
Same as `activate_env.ps1`: opens a new shell with the virtual environment activated.

### run_coverage.sh
Same as `run_coverage.ps1`: runs pytest with coverage for the configured tests.

**To run the tests (from the repo root):**

```
./scripts/setup_env.sh
./scripts/activate_env.sh
./scripts/run_coverage.sh
```

## coverage report output

A new HTML coverage report is written to:

```
tests/coverage_report_html/index.html
```

# download_dataset.py

Run this to get the hugging face dataset.
