ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.03-py3

FROM ${FROM_IMAGE_NAME} as base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.8.3 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# `builder-base` stage is used to build deps + create our virtual environment
FROM base as builder-base
RUN apt-get update && apt-get install --no-install-recommends -y \
    # deps for installing poetry
    curl \
    # deps for building python deps
    build-essential

RUN curl -sSL https://install.python-poetry.org | python -

WORKDIR $PYSETUP_PATH
COPY  pyproject.toml ./

RUN poetry install --without dev --without gpu --without test  --no-interaction --no-ansi


FROM builder-base as development
WORKDIR $PYSETUP_PATH

RUN mkdir /var/logs

COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

RUN poetry install --without dev --without gpu --without test  --no-interaction --no-ansi

RUN apt update && \
    apt install -y  netcat-traditional

WORKDIR /src
RUN mkdir /src/models






FROM builder-base as development-gpu
RUN mkdir /var/logs


WORKDIR $PYSETUP_PATH

COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

RUN poetry install --without dev  --without test  --no-interaction --no-ansi

WORKDIR /src
RUN mkdir /src/models


RUN apt update && \
    apt install -y  netcat-traditional



