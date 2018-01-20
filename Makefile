PIP := .env/bin/pip
PYTHON := .env/bin/python3

# create virtual environment
.env:
	virtualenv .env -p python3

# install dev dependencies
develop: .env
	$(PIP) install -U -r requirements.txt

serve: develop
	$(PYTHON) app.py

run:
	docker build -t travelbid . && \
	docker run -p 5000:5000 \
	-it python3 app.py
		