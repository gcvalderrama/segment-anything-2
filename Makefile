setup:
	python -m venv .venv

dependencies:
	pip install -e .
	pip install -e ".[demo]"
	pip install -r requirements.txt
	pip uninstall torch
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

	