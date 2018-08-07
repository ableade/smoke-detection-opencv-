#!/bin/bash
#Resizes images in a target directory to a standard size for our neural network
dir=$1/*.jpg
echo $dir
for i in $dir; do
    printf "Resize $i\n"
    convert "$i" -resize 300x300\! "$i"
done