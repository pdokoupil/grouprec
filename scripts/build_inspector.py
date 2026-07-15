#!/usr/bin/env python
"""Thin wrapper -- the inspector generator now lives in the installed package.

After ``pip install grouprec[torch]`` run ``grouprec-build-inspector`` (console
script) or ``python -m grouprec.inspector.build``. This wrapper keeps
``python scripts/build_inspector.py`` working from a repo checkout.
"""

from grouprec.inspector.build import main

if __name__ == "__main__":
    main()
