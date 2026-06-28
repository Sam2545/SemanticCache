.PHONY: up down logs test

up:
	./start.sh

down:
	docker compose down

logs:
	docker compose logs -f

test:
	pytest -m "not integration"
