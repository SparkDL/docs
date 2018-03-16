#!/bin/bash

mkdocs build &&
cd site
git add -A
git commit -m "update site"
git push
