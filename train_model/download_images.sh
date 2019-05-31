#!/bin/bash

rm -rf ./downloads/

googleimagesdownload -k "art OR painting OR sculpture OR exhibition" -f jpg -i art -l 300 -cd /usr/lib/chromium-browser/chromedriver
googleimagesdownload -k "trash OR rubbish" -f jpg -i trash -l 300 -cd /usr/lib/chromium-browser/chromedriver

mogrify -resize 224x224\! -colorspace sRGB -format JPG ./downloads/*/*.*
rm ./downloads/*/*.jpg
