#!/usr/bin/env python3
import sys
from code_mia.cli import main

raise SystemExit(main(["download", *sys.argv[1:]]))

