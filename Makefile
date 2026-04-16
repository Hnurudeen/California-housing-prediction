.PHONY: run install develop clean lint

## Run the full analysis pipeline
run:
	python run_analysis.py

## Install dependencies
install:
	pip install -r requirements.txt

## Install the package in development mode (enables `import california_housing` from anywhere)
develop:
	pip install -e .

## Lint source code with flake8
lint:
	flake8 california_housing/ run_analysis.py

## Delete compiled Python files and cache directories
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
