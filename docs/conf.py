extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
source_suffix = ".rst"
autosummary_generate = True

master_doc = "index"
project = "BMDS"
copyright = "2022, BMDS Python Team"
author = "BMDS Python Team"

language = None
exclude_patterns = []
pygments_style = "sphinx"

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- Options for HTML output ----------------------------------------------
html_theme = "alabaster"
html_theme_options = {
    "show_powered_by": False,
    "description": "A Python API to US EPA's Benchmark Dose Modeling Software (BMDS)",
    "github_user": "shapiromatron",
    "github_repo": "bmds",
    "github_count": False,
    "show_related": False,
    "sidebar_includehidden": False,
}
html_static_path = ["_static"]
html_sidebars = {"**": ["about.html", "navigation.html", "relations.html", "searchbox.html"]}
