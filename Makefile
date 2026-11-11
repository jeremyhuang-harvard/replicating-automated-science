.PHONY: results check test setup

results:  ## rebuild results/ from the saved outputs (no key, no model)
	python3 analysis/make_results.py

check: results  ## fail if the committed results differ from a fresh build
	git diff --exit-code -- results/

test:  ## root checks, then each case study's original test suite
	uv run --group dev pytest -q tests
	cd case-studies/expected-parrot && uv run python -m unittest discover
	cd case-studies/automated-social-science && uv run --group dev pytest -q

setup:  ## install the dev tools; reruns also need mlx-lm (see README)
	uv sync
