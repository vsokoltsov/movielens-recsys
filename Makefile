install-kernel:
	uv run python -m ipykernel install --user --name=movielens-recsys --display-name="Recommendation system (MovieLens)"

mypy:
	uv run mypy pipelines/ tests/

black:
	uv run black --check pipelines/ tests/

black-fix:
	uv run black pipelines/ tests/

ruff:
	uv run ruff check pipelines/ tests/ --fix

lint:
	make mypy & make black-fix & make ruff