.PHONY: dev pre-commit isort black mypy flake8 pylint lint

dev: pre-commit

pre-commit:
	pre-commit install
	pre-commit autoupdate

isort:
	isort . --profile black

flake8:
	flake8  triton_server

black:
	black .

mypy:
	mypy -p triton_server

pylint:
	pylint triton_server


lint: isort black mypy  pylint flake8

check_and_rename_env:
	  @if [ -e ".env" ]; then \
        echo "env file exists."; \
      else \
      	cp .env.example .env | \
        echo "File does not exist."; \
      fi


build:check_and_rename_env
	docker compose build

build_docker_gpu:check_and_rename_env
	docker compose -f docker-compose.gpu.yml build


run_gpu:
	docker compose -f docker-compose.gpu.yml up

gpu_stop:
	docker compose -f docker-compose.gpu.yml down
