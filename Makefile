.DEFAULT_GOAL := all
.PHONY: all check verify comparisons audit build pdf freeze
all check verify comparisons audit build pdf freeze:
	+$(MAKE) -C paper $@
