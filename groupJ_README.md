# environment scrips in the scripts directory

## setup_env.ps1
creates the virtual environment and installs depenencies. 

## activate_env.ps1
activates the environment

## run_coverage.ps1
runs the tests in the tests directory and outputs the coverage report in the test directory.
you have to add new tests to this script.

# to run the tests
```
.\scripts\setup_env.ps1
.\scripts\activate_env.ps1
.\scripts\run_coverage.ps1
```
## a new coverage report is written to:
```
.\tests\coverage_report_html\index.html
```

# download_dataset.py
run this to get the hugging face dataset