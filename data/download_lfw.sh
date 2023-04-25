#!/bin/bash

# Download the LFW dataset
curl -O http://vis-www.cs.umass.edu/lfw/lfw.tgz

# Extract the contents of the compressed file
tar -xzvf lfw.tgz

# Rename the extracted folder to 'lfw_original'
mv lfw lfw_original

# Remove the compressed file
rm lfw.tgz
