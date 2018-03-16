#!/bin/bash

git add -A && git commit -m "update documents" && git push

mkdocs build && \
cd site && \
git add -A && \
git commit -m "update site" && \
git push
