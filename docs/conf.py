# -*- coding: utf-8 -*-
#
# pyramid documentation build configuration file, created by
# sphinx-quickstart on Wed Jul 16 13:18:14 2008.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.


import datetime
import inspect
import os
import sys
import warnings

import pkg_resources
from docutils import nodes
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.text import TextTranslator

sys.path.insert(0, os.path.abspath('..'))

import hea

warnings.simplefilter('ignore', DeprecationWarning)


# skip raw nodes


def raw(*arg):
    raise nodes.SkipNode


TextTranslator.visit_raw = raw


# make sure :app: doesn't mess up LaTeX rendering
def nothing(*arg):
    pass


LaTeXTranslator.visit_inline = nothing
LaTeXTranslator.depart_inline = nothing

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'repoze.sphinx.autointerface',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinxcontrib.autoprogram',
    # enable pylons_sphinx_latesturl when this branch is no longer "latest"
    # 'pylons_sphinx_latesturl',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General substitutions.
project = hea.__name__
thisyear = datetime.datetime.now().year
author = hea.__author__
copyright = '{}, {} (National Reserved Council Canada'.format(thisyear, author)

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
# version = pkg_resources.get_distribution('pyHEA').version
version = hea.__version__
# The full version, including alpha/beta/rc tags.
release = version

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_themes/README.rst',
]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = book and 'bw' or 'tango'
pygments_style = 'sphinx'

html_theme = 'alabaster'
# html_theme_options = dict(
#     github_url='https://github.com/pyHEA',
#     # On master branch and new branch still in
#     # pre-release status: true; else: false.
#     in_progress='true',
#     # On branches previous to "latest": true; else: false.
#     outdated='false',
# )

# Control display of sidebars
html_sidebars = {
    '**': [
        'localtoc.html',
        'ethicalads.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'pyHEA v{}'.format(release)

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# Do not use smart quotes.
smartquotes = False

# Output file base name for HTML help builder.
htmlhelp_basename = '{}doc'.format(project)

# Options for LaTeX output
# ------------------------

latex_engine = 'xelatex'
latex_use_xindy = False

# The paper size ('letter' or 'a4').
latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
    ('latexindex', 'pyhea.tex', 'pyHEA Documentation', 'Conrard TETSASI', 'manual'),
]

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_toplevel_sectioning = "section"

# If false, no module index is generated.
latex_domain_indices = False

latex_elements = {
    'releasename': 'Version',
    'title': r'pyHEA',
    #    'pointsize':'12pt', # uncomment for 12pt version
}

# For a list of all settings, visit http://sphinx-doc.org/config.html

# -- Options for linkcheck builder -------------------------------------------

# List of items to ignore when running linkcheck
linkcheck_ignore = [
    r'http://localhost:\d+',
    r'http://localhost',
    r'https://webchat.freenode.net/#pHEA',  # JavaScript "anchor"
]
