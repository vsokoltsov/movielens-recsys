install-kernel:
	uv run python -m ipykernel install --user --name=movielens-recsys --display-name="Recommendation system (MovieLens)"

mypy:
	uv run mypy recsys/

black:
	uv run black --check recsys/

black-fix:
	uv run black recsys/

ruff:
	uv run ruff check recsys/ --fix

lint:
	make mypy & make black-fix & make ruff

api:
	docker-compose up api

jupyter:
	docker-compose up jupyter

train-cli:
	uv run python -m recsys.modeling.train --model-type=$(model_type) --bucket-name=$(bucket_name)

predict-cli:
	uv run python -m recsys.modeling.predict $(user_id) --model-type=$(model_type) --bucket-name=$(bucket_name)