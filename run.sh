#!/usr/bin/env sh

CURR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Attempt to run the main script
python main.py

if [ $? -eq 255 ]
then
    # Revertion was successful, run command again
    python main.py
fi

git checkout $CURR_BRANCH
