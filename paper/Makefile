.PHONY: all check verify comparisons audit build pdf freeze
PYTHON ?= python3

all: check
	$(MAKE) build

check: verify
	$(MAKE) comparisons

verify:
	@mkdir -p outputs
	$(PYTHON) -B verify_low.py > outputs/low-checks.txt
	$(PYTHON) -B verify_bbjt.py > outputs/bbjt-quartic.txt
	$(PYTHON) -B fixed_six.py > outputs/fixed-six.txt
	$(PYTHON) -B verify_fixed.py > outputs/fixed-small.txt
	$(PYTHON) -B verify_paper.py > outputs/analytic-checks.txt
	$(PYTHON) -B verify_transfer.py --core
	$(PYTHON) -B -m research.verify_term_bounds > outputs/center-checks.txt
	$(PYTHON) -B low_inputs.py > outputs/collected-small.json
	$(PYTHON) -B evidence/check_exports.py > outputs/export-checks.txt
	$(PYTHON) -B evidence/check_primitive_fields.py > outputs/primitive-checks.txt
	$(PYTHON) -B evidence/check_factor_roundtrip.py
	$(PYTHON) -B evidence/check_grid.py > outputs/grid-check.txt
	$(PYTHON) -B evidence/check_table.py > outputs/table-replay.txt
	$(PYTHON) -B evidence/check_stationary_jets.py
	$(PYTHON) -B evidence/check_quintic_support.py > outputs/quintic-scope.txt
	$(PYTHON) -B evidence/check_fold_figure.py
	$(PYTHON) -B finite.py --stages 47 48 257 261 512 > outputs/finite-boundaries.jsonl

comparisons:
	@mkdir -p outputs
	$(PYTHON) -B evidence/check_seed_strips.py > outputs/seed-strips.txt
	$(PYTHON) -B archive_comparison.py > outputs/archive-checks.txt
	$(PYTHON) -B verify_transfer.py > outputs/transfer-checks.txt
	$(PYTHON) -B evidence/check_renewal_support.py > outputs/renewal-support.txt
	$(PYTHON) -B evidence/check_mechanical_support.py > outputs/mechanical-comparison.txt
	$(PYTHON) -B evidence/check_budget_comparisons.py > outputs/budget-comparisons.txt
	$(PYTHON) -B evidence/check_literature_fields.py > outputs/literature-fields.txt

audit: comparisons

build:
	$(PYTHON) -B build.py
	$(PYTHON) -B evidence/check_pdfs.py

pdf:
	$(PYTHON) -B build.py

freeze: build
	$(PYTHON) -B bundle_replay.py --release-manifest
	sha256sum --strict -c SHA256SUMS
