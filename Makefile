.PHONY: test offline-test
test:
	python -m pytest -q
offline-test:
	unshare -Urn python -m pytest -q
