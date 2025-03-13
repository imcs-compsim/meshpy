# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
"""Check that the MeshPy internal imports adhere to our coding guidelines."""

import ast
import sys
from pathlib import Path


def get_import_details_from_code(code):
    """Parses Python code and extracts imported modules, explicitly imported
    names, and their aliases.

    Args:
        code (str): Python source code as a string.

    Returns:
        dict: Dictionary where:
            - Keys are module names.
            - Values are:
                - None for normal imports (e.g., `import os`).
                - A list of tuples for `from module import ...` with (original name, alias or None).
    """

    tree = ast.parse(code)
    imports = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Store module name with alias if it exists
                imports[alias.name] = alias.asname if alias.asname else None
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Store explicitly imported names with aliases
                imports[node.module] = [
                    (alias.name, alias.asname if alias.asname else None)
                    for alias in node.names
                ]

    return imports


def check_imports(file_path):
    """Reads a Python file and checks if the imports align with the MeshPy
    coding guidelines."""
    with open(file_path, "r") as f:
        code = f.read()

    imports = get_import_details_from_code(code)

    return_value = True
    for module, explicit_imports in imports.items():
        if module.startswith("meshpy"):
            if explicit_imports is None:
                # We don't allow direct imports of MeshPy, e.g., import meshpy.core
                print(f"Don't directly import MeshPy modules: {module}")
                return_value = False
            elif isinstance(explicit_imports, list):
                for name, alias in explicit_imports:
                    if alias is not None:
                        if not f"_{name}" == alias:
                            # We require MeshPy import aliases to have a leading underscore and then the
                            # same name as the original object, e.g., from meshpy.core.mesh import Mesh as _Mesh
                            print(
                                f"The name of the MeshPy alias does not match the original name: from {module} import {name} as {alias}"
                            )
                            return_value = False
                    else:
                        # We don't imports of MeshPy objects without an alias, e.g., from meshpy.core.mesh import Mesh
                        print(
                            f"MeshPy imports have to have an alias (with a leading underscore): from {module} import {name}"
                        )
                        return_value = False
            else:
                # We allow this as it helps for some type hints
                if module == "meshpy.core.conf":
                    continue
                print(
                    f"Don't directly import MeshPy modules: {module} as {explicit_imports}"
                )
                return_value = False
    return return_value


def main():
    """Runs check_imports on all staged Python files."""

    files_to_check = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]

    if not files_to_check:
        return 0

    exit_code = 0
    for file_path in files_to_check:
        print(f"\nChecking imports in: {file_path}")
        if not check_imports(file_path):
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    """Execution part of script."""
    main()
