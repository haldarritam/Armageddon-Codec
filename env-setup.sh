#!/usr/bin/env bash

echo "Configuring environment for Python development"

conda env create -f environment.yml

read -p "Press enter to continue"
