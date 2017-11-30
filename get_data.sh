#!/usr/bin/env bash

if [ ! -d train ]; then
  mkdir train
  cd train
  wget "https://www.cs.helsinki.fi/u/jgpyykko/reuters.zip"
  unzip reuters.zip
  rm reuters.zip
else
  cd train
fi;

cd REUTERS_CORPUS_2
unzip codes.zip -d codes
unzip dtds.zip -d dtds
rm dtds.zip codes.zip

unzip '*.zip' -d data
rm *.zip
