#!/usr/bin/env python3
"""Sphinx configuration for PyMC-Marketing Docs."""

import multiprocessing
import os

import pymc_marketing  # isort:skip

# -- Fast Build Mode Configuration ----------------------------------------
# Set environment variables to speed up builds during development:
#   - PYMC_MARKETING_FAST_DOCS=1: Skip notebooks and heavy API generation
#   - SKIP_NOTEBOOKS=1: Skip notebook execution only
#   - SKIP_API_GENERATION=1: Skip API documentation generation only

FAST_DOCS = os.environ.get("PYMC_MARKETING_FAST_DOCS", "0") == "1"
SKIP_NOTEBOOKS = os.environ.get("SKIP_NOTEBOOKS", "0") == "1" or FAST_DOCS
SKIP_API_GENERATION = os.environ.get("SKIP_API_GENERATION", "0") == "1" or FAST_DOCS

if FAST_DOCS:
    print("=" * 70)
    print("FAST BUILD MODE ENABLED")
    print("  - Notebooks: SKIPPED")
    print("  - API Generation: SKIPPED")
    print("  - Build time: ~30-60 seconds")
    print("=" * 70)
elif SKIP_NOTEBOOKS or SKIP_API_GENERATION:
    print("=" * 70)
    print("PARTIAL FAST BUILD MODE")
    print(f"  - Notebooks: {'SKIPPED' if SKIP_NOTEBOOKS else 'ENABLED'}")
    print(f"  - API Generation: {'SKIPPED' if SKIP_API_GENERATION else 'ENABLED'}")
    print("=" * 70)

# -- General configuration ------------------------------------------------

# General information about the project.
project = "pymc-marketing"
author = "PyMC Labs"
copyright = f"2022-%Y, {author}"
html_title = "Open Source Marketing Analytics Solution"

# The master toctree document.
master_doc = "index"

# Add any Sphinx extension module names here, as strings
extensions = [
    # extensions from sphinx base
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    # extensions provided by other packages
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",  # needed to plot in docstrings
    "myst_nb",
    "notfound.extension",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_remove_toctrees",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

# The full version, including alpha/beta/rc tags.
release = pymc_marketing.__version__

# The version info for the project you're documenting
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if rtd_version.lower() == "stable":
        version = release.split("+")[0]
    elif rtd_version.lower() == "latest":
        version = "dev"
    else:
        version = rtd_version
else:
    version = "local"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "build",
    "jupyter_execute",
    "jupyter_cache",
    "**.ipynb_checkpoints",
]

# Fast build mode: Skip notebooks
if SKIP_NOTEBOOKS:
    exclude_patterns.extend(
        [
            "notebooks/**",
            "guide/benefits/model_deployment.ipynb",
        ]
    )
    print("  ⚡ Excluding all notebooks from build")

# Fast build mode: Skip API generation
if SKIP_API_GENERATION:
    exclude_patterns.extend(
        [
            "api/generated/**",
        ]
    )
    print("  ⚡ Excluding API documentation from build")

# The reST default role (used for this markup: `text`) to use for all documents.
# This sets the behaviour to be the same as in markdown
default_role = "code"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# sphinx settings related to generation of translatable sources
gettext_uuid = True
gettext_compact = False
locale_dirs = ["../../locales"]

# -- Extension configuration ------------------------------------------------

# configure notfound extension to not add any prefix to the urls
notfound_urls_prefix = "/en/latest/"

# exclude method pages from toctree to make pages lighter and build faster
remove_from_toctrees = ["**/classmethods/*"]

# myst config
# Use cache mode for faster subsequent builds (only re-executes modified notebooks)
# Use "off" in fast build mode to skip all notebook execution
if SKIP_NOTEBOOKS:
    nb_execution_mode = "off"
else:
    nb_execution_mode = "cache"  # Changed from "auto" for better performance

nb_execution_cache_path = ".jupyter_cache"  # Persistent cache directory
nb_execution_timeout = 600  # 10 minutes per notebook
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
nb_execution_excludepatterns = [
    # Heavy notebooks that take too long - execute manually when needed
    "notebooks/mmm/mmm_case_study.ipynb",
    "notebooks/mmm/mmm_multidimensional_example.ipynb",
    "notebooks/mmm/mmm_tvp_example.ipynb",
    "notebooks/mmm/mmm_time_varying_media_example.ipynb",
    "notebooks/clv/dev/*.ipynb",  # Development notebooks
]
nb_kernel_rgx_aliases = {".*": "python3"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 0

# numpydoc and autodoc typehints config
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# Enable parallel builds for faster processing
# Use all CPUs except one to keep system responsive
autodoc_parallel = max(1, multiprocessing.cpu_count() - 1)

# Optimize autosummary generation
autosummary_generate = True
# Don't regenerate unchanged files (speeds up incremental builds)
autosummary_generate_overwrite = False
# fmt: off
numpydoc_xref_ignore = {
    "of", "or", "optional", "default", "numeric", "type", "scalar", "1D", "2D", "3D", "nD", "array",
    "instance", "M", "N"
}
# fmt: on
numpydoc_xref_aliases = {
    "TensorVariable": ":class:`~pytensor.tensor.TensorVariable`",
    "RandomVariable": ":class:`~pytensor.tensor.random.RandomVariable`",
    "ndarray": ":class:`~numpy.ndarray`",
    "InferenceData": ":class:`~arviz.InferenceData`",
    "Model": ":class:`~pymc.Model`",
    "tensor_like": ":term:`tensor_like`",
    "unnamed_distribution": ":term:`unnamed_distribution`",
}
# don't add a return type section, use standard return with type info
typehints_document_rtype = False

# intersphinx configuration to ease linking arviz docs
intersphinx_mapping = {
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "examples": ("https://www.pymc.io/projects/examples/en/latest/", None),
    "mpl": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# Cache intersphinx inventories for faster builds
intersphinx_cache_limit = 10  # Days to cache
intersphinx_timeout = 30  # Seconds

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "labs_sphinx_theme"

html_favicon = "_static/favicon.ico"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "logo": {
        "image_light": "flat_logo.png",
        "image_dark": "flat_logo_darkmode.png",
    },
    "analytics": {"google_analytics_id": "G-DNPNG22HVY"},
}
html_context = {
    "github_user": "pymc-labs",
    "github_repo": "pymc-marketing",
    "github_version": "main",
    "doc_path": "docs/source/",
    "default_mode": "light",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static/"]
html_css_files = ["custom.css"]

# -- Options for LaTeX output ---------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "pymc_marketing.tex", "pymc_marketing Documentation", author, "manual")
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "pymc_marketing", "pymc_marketing Documentation", [author], 1)
]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pymc_marketing",
        "pymc_marketing Documentation",
        author,
        "pymc_marketing",
        "Bayesian MMMs and CLVs in PyMC.",
        "Miscellaneous",
    )
]
