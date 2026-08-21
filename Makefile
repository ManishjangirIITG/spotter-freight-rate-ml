.PHONY: setup train evaluate predict test clean

setup:
	pip install -e .

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

predict:
	python scripts/predict.py

test:
	pytest -v

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -r {} +