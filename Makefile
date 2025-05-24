run:
	poetry run python src/cli_app.py

fmt:
	poetry run black .
	poetry run isort .