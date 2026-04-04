#!/usr/bin/env python3
"""Sphinx configuration for PyMC-Marketing Docs."""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pymc_marketing  # isort:skip

# -- General configuration ------------------------------------------------

# General information about the project.
project = "PyMC-Marketing"
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
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    # extensions provided by other packages
    "sphinx_autodoc_typehints",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",  # needed to plot in docstrings
    "myst_nb",
    "notfound.extension",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_remove_toctrees",
    "sphinx_sitemap",
    "sphinxext.opengraph",
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
    rtd_version = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "build",
    "jupyter_execute",
    "jupyter_cache",
    "**.ipynb_checkpoints",
]

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

# exclude method pages from toctree to make pages lighter and build faster
remove_from_toctrees = ["**/classmethods/*"]

# myst config
nb_execution_mode = "auto"
nb_execution_excludepatterns = ["*.ipynb"]
nb_kernel_rgx_aliases = {".*": "python3"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 0

# numpydoc and autodoc typehints config
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
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


# linkcode extension (links of [source] pointing to github)
def linkcode_resolve(domain, info):
    """Given sphinx contextual objects when building the docs, generate links to source on GH."""

    def find_obj() -> object:
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        return obj

    def find_source(obj):
        fn = Path(inspect.getsourcefile(obj))
        fn = fn.relative_to(Path(pymc_marketing.__file__).parent)
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    def fallback_source():
        return info["module"].replace(".", "/") + ".py"

    if domain != "py" or not info["module"]:
        return None

    try:
        obj = find_obj()
    except Exception:
        filename = fallback_source()
    else:
        try:
            path, start_line, end_line = find_source(obj)
            filename = f"pymc_marketing/{path}#L{start_line}-L{end_line}"
        except Exception:
            try:
                filename = obj.__module__.replace(".", "/") + ".py"
            except AttributeError:
                # Some objects do not have a __module__ attribute (?)
                filename = fallback_source()

    tag = subprocess.Popen(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).communicate()[0][:-1]
    return f"https://github.com/pymc-labs/pymc-marketing/blob/{tag}/{filename}"


# -- HTML specific extensions -------------------------------------

# configure notfound extension
notfound_urls_prefix = "/en/latest/"

# opengraph metadata settings
ogp_site_url = "https://www.pymc-marketing.io/en/stable/"
ogp_canonical_url = "https://www.pymc-marketing.io/en/stable/"
ogp_image = "https://www.pymc-marketing.io/en/stable/_images/marketing-logo-light.jpg"
ogp_enable_meta_description = False


# sitemap extension configuration
site_url = "https://www.pymc-marketing.io/"
sitemap_url_scheme = f"{{lang}}{rtd_version}/{{link}}"


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "labs_sphinx_theme"
html_extra_path = ["robots.txt"]
html_copy_source = (
    False  # don't include rst source files as _sources/...txt in the build
)

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
    "baseurl": "https://www.pymc-marketing.io/",
    "rtd_version": rtd_version,
    "translations": ["en", "es"],
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
