.PHONY: venv, venvd, poetry


##### REPOSITORY & VENV #####
poetry:
	curl -sSL https://install.python-poetry.org | python3.9 -

venv:
	poetry config virtualenvs.in-project true
	python3.9 -m venv .venv; \
	cp .env_tmpl .env; \
	echo "set -a && . ./.env && set +a" >> .venv/bin/activate; \
	. .venv/bin/activate; \
	pip install -U pip setuptools wheel; \
	poetry install

venvd:
	rm -rf .venv
