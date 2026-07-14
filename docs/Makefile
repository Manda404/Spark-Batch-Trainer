# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SPHINXAUTO    ?= sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = build
PYTHON        ?= python

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Project shortcuts:"
	@echo "  make html      Build the HTML documentation"
	@echo "  make clean     Remove the build directory"
	@echo "  make rebuild   Clean, then rebuild the HTML documentation"
	@echo "  make open      Open the built HTML documentation in a browser"
	@echo "  make livehtml  Rebuild and serve with live-reload (requires sphinx-autobuild)"
	@echo "  make linkcheck Check all external links for validity"
	@echo "  make doctest   Run doctest snippets embedded in the documentation"

.PHONY: help Makefile html clean rebuild open livehtml linkcheck doctest

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

rebuild: clean html

open:
	@$(PYTHON) -m webbrowser -t "file://$(CURDIR)/$(BUILDDIR)/html/index.html"

livehtml:
	@$(SPHINXAUTO) "$(SOURCEDIR)" "$(BUILDDIR)/html" --open-browser

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doctest:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route any other Sphinx builder (epub, latex, man, ...)
# to Sphinx using the "make mode" option. $(O) is a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
