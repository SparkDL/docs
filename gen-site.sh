#!/bin/bash

git add -A && git commit -m "update documents"

mkdocs build && \
cd site && \
git add -A && \
git commit -m "update site" && \
git push
