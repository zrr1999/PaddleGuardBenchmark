install:
    uv sync --all-extras --dev

install-tuna:
    uv sync --all-extras --dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple

ruff:
    uv run ruff check . --fix --unsafe-fixes
    uv run ruff format .

pyright:
    uv run pyright .

test:
    uv run pytest

check:
    just ruff
    just pyright

push:
    just test
    git push
    git push --tags
