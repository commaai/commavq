# Set the execution policy if needed, though running as a simple script
# This script runs pytest with coverage on the utils.vqvae module

# Using python -m pytest ensures the current directory is securely added to the python path
$env:PYTHONPATH = ".;$env:PYTHONPATH"
python -m pytest tests/test_compressor_config.py --cov=utils.vqvae --cov-report=term-missing --cov-report=html:tests/coverage_report_html --cov-report=xml:tests/coverage.xml
