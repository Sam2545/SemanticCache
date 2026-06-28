.PHONY: up down logs test serve

up:
	./start.sh

serve:
	./start-local.sh

down:
	docker compose down

logs:
	docker compose logs -f

test:
	pytest -m "not integration"
