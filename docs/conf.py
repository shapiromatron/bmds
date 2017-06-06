# -*- coding: utf-8 -*-

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
source_suffix = '.rst'
autosummary_generate = True

master_doc = 'index'
project = u'BMDS'
copyright = u'2017, Andy Shapiro'
author = u'Andy Shapiro'

language = None
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = False

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- Options for HTML output ----------------------------------------------
html_theme = 'alabaster'
html_theme_options = {
    'show_powered_by': False,
    'description': 'A Python API to EPA\'s BMDS software',
    'github_user': 'shapiromatron',
    'github_repo': 'bmds',
    'github_count': False,
    'show_related': False,
    'sidebar_includehidden': False
}
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}
