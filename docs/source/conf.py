# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import importlib
import os
import sys
import textwrap
import warnings

import packaging

# Ignore FutureWarnings coming from docutils remove this once we can upgrade to Sphinx 5.0
# https://github.com/sphinx-doc/sphinx/issues/9777
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get the morpheus root from the environment variable or default to finding it relative to this file
morpheus_root = os.environ.get('MORPHEUS_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Make sure we can access the digital fingerprinting example
sys.path.append(os.path.join(morpheus_root, 'examples/digital_fingerprinting/production/morpheus'))

# Add the Sphinx extensions directory to sys.path to allow for the github_link extension to be found
sys.path.insert(0, os.path.abspath('sphinxext'))

from github_link import make_linkcode_resolve  # noqa

# Set an environment variable we can use to determine if we are building docs
os.environ["MORPHEUS_IN_SPHINX_BUILD"] = "1"

# -- Project information -----------------------------------------------------

project = 'morpheus'
copyright = '2024, NVIDIA'
author = 'NVIDIA'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# Load the _version file according to https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
spec = importlib.util.spec_from_file_location("_version", "../../python/morpheus/morpheus/_version.py")
module = importlib.util.module_from_spec(spec)
sys.modules["_version"] = module
spec.loader.exec_module(module)

# The full version including a/b/rc tags
release = sys.modules["_version"].get_versions()["version"]

version_obj = packaging.version.parse(release)

# The short version
version = f"{version_obj.major:02d}.{version_obj.minor:02d}"

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'exhale',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'myst_parser',
    'nbsphinx',
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
]

# Breathe Configuration
breathe_default_project = "morpheus"

# This will be set when invoked by cmake
build_dir = os.environ.get('BUILD_DIR', './')
doxygen_tmp_dir = os.path.join(build_dir, "_doxygen/xml")
breathe_projects = {"morpheus": doxygen_tmp_dir}

exhale_args = {
    "containmentFolder":
        "./_lib",
    "rootFileName":
        "index.rst",
    "doxygenStripFromPath":
        "../../",
    "rootFileTitle":
        "C++ API",
    "createTreeView":
        True,
    "exhaleExecutesDoxygen":
        True,
    "exhaleDoxygenStdin":
        textwrap.dedent('''
        BRIEF_MEMBER_DESC = YES
        BUILTIN_STL_SUPPORT = YES
        DOT_IMAGE_FORMAT = svg
        EXCLUDE_PATTERNS = */tests/* */include/nvtext/* */__pycache__/* */doca/*
        EXCLUDE_SYMBOLS = "@*" "cudf*" "py::literals" "RdKafka" "mrc*" "std*" "PYBIND11_NAMESPACE*"
        EXTENSION_MAPPING = cu=C++ cuh=C++
        EXTRACT_ALL = YES
        FILE_PATTERNS = *.c *.cc *.cpp *.h *.hpp *.cu *.cuh *.md
        HAVE_DOT = YES
        HIDE_UNDOC_MEMBERS = NO
        INPUT = ../../python/morpheus/morpheus/_lib ../../python/morpheus_llm/morpheus_llm/_lib
        INTERACTIVE_SVG = YES
        SOURCE_BROWSER = YES
        ENABLE_PREPROCESSING = YES
        MACRO_EXPANSION = YES
        EXPAND_ONLY_PREDEF = NO
        PREDEFINED = "MORPHEUS_EXPORT=" \
                     "DOXYGEN_SHOULD_SKIP_THIS=1"
    ''')
}

# Include Python objects as they appear in source files
# Default: alphabetically ('alphabetical')
# autodoc_member_order = 'groupwise'
# Default flags used by autodoc directives
# autodoc_default_options = {
#     'members': True,
#     'show-inheritance': True,
# }

autosummary_imported_members = False
autosummary_generate = True  # Generate autodoc stubs with summaries from code
autoclass_content = "class"  # Dont show __init__
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
autodoc_typehints = "description"  # Sphinx-native method. Not as good as sphinx_autodoc_typehints
autodoc_typehints_description_target = "documented"  # Dont double up on type hints
add_module_names = False  # Remove namespaces from class/method signatures
myst_heading_anchors = 4  # Generate links for markdown headers
autodoc_mock_imports = [
    "cudf",  # Avoid loading GPU libraries during the documentation build
    "cupy",  # Avoid loading GPU libraries during the documentation build
    "databricks.connect",
    "datacompy",
    "langchain",
    "langchain_core",
    "morpheus.cli.commands",  # Dont document the CLI in Sphinx
    "pandas",
    "pydantic",
    "pymilvus",
    "tensorrt",
    "torch",
    "tqdm"
]

suppress_warnings = [
    "myst.header"  # Allow header increases from h2 to h4 (skipping h3)
]

# Config numpydoc
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False

# Config linkcheck
# Ignore localhost and url prefix fragments
# Ignore openai.com links, as these always report a 403 when requested by the linkcheck agent
# The way Github handles anchors into markdown files is not compatible with the way linkcheck handles them.
# This allows us to continue to verify that the links are valid, but ignore the anchors.
linkcheck_ignore = [
    r'http://localhost:\d+/',
    r'https://localhost:\d+/',
    r'^http://$',
    r'^https://$',
    r'https://(platform\.)?openai.com',
    r'https://code.visualstudio.com',
    r"^https://github.com/nv-morpheus/Morpheus/blob/.*#.+$"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_logo = '_static/main_nv_logo_square.png'

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'style_nav_header_background': '#000000',  # Toc options
    'collapse_navigation': False,
    'navigation_depth': 6,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_js_files = ["example_mod.js"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'morpheusdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'morpheus.tex', 'Morpheus Documentation', 'NVIDIA', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'morpheus', 'Morpheus Documentation', [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc,
     'morpheus',
     'Morpheus Documentation',
     author,
     'morpheus',
     'One line description of project.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ('https://docs.python.org/', None), "scipy": ('https://docs.scipy.org/doc/scipy/reference', None)
}


def setup(app):
    app.add_css_file('omni-style.css')
    app.add_css_file('copybutton.css')
    app.add_css_file('infoboxes.css')
    app.add_css_file('params.css')
    app.add_css_file('references.css')
    app.add_css_file('py_properties.css')


# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    'morpheus', 'https://github.com/nv-morpheus/Morpheus'
    '/blob/{revision}/'
    '{package}/{path}#L{lineno}')

# Set the default role for interpreted code (anything surrounded in `single
# backticks`) to be a python object. See
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role
default_role = "py:obj"
