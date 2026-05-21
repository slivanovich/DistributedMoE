#!/bin/bash

find . -type fd -name ".DS_Store" -exec rm -rf {} \;
find src -name "__pycache__" -exec rm -rf {} \;
