.PHONY: install all clean-tex help
.DEFAULT_GOAL := help


help:
	@echo "Makefile help: (Tested on Linux)"
	@echo "* install	to install the requirements into your current virtaul env"

install: | package-install download

package-install:
	python -m pip install --no-cache -r requirements.txt

clean-py:
	find . \( -name __pycache__ -o -name "*.pyc" \) -delete
