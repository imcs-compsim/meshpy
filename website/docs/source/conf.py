# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# general configuration
project = "BeamMe"
copyright = "2025, BeamMe Authors"
author = "BeamMe Authors"

# html theme configuration
html_theme = "pydata_sphinx_theme"
html_title = "BeamMe"
html_theme_options = {
    "github_url": "https://github.com/beamme-py/beamme",
}
html_show_sourcelink = False  # Hide "View Source" link on the right side of the page

# extensions
extensions = [
    "myst_parser",  # to enable Markdown support
    "sphinxcontrib.jquery",  # to enable custom JavaScript (open links in new empty tab)
]

# markdown configuration
myst_enable_extensions = [
    "colon_fence",  # For ::: fenced code blocks
    "linkify",  # Auto-detects URLs and makes them hyperlinks
]
myst_heading_anchors = 3  # automatic heading anchors for Markdown files

# JavaScript configuration (to open links in new tabs)
html_js_files = ["js/custom.js"]
html_static_path = ["static"]
