#!/bin/bash

rm -rf ./downloads/

googleimagesdownload -k "art OR painting OR sculpture OR exhibition OR exhibit OR design OR kunstwerk" -f jpg -i art -l 3000 -cd /usr/lib/chromium-browser/chromedriver
googleimagesdownload -k "trash OR rubbish OR junk OR abfall OR müll OR waste OR garbage OR scrap" -f jpg -i trash -l 3000 -cd /usr/lib/chromium-browser/chromedriver

mogrify -resize 224x224\! -colorspace sRGB -format JPG ./downloads/*/*.*
rm ./downloads/*/*.jpg
