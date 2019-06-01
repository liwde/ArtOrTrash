#!/bin/bash

rm -rf ./downloads/

parallel -j 0 googleimagesdownload -k {} -f jpg -i art -l 500 -cd /usr/lib/chromium-browser/chromedriver ::: "art" "painting" "sculpture" "exhibition" "exhibit" "design" "kunstwerk"
parallel -j 0 googleimagesdownload -k {} -f jpg -i trash -l 500 -cd /usr/lib/chromium-browser/chromedriver ::: "trash" "rubbish" "junk" "waste" "garbage" "scrap" "abfall"

mogrify -resize 224x224\! -colorspace sRGB -format JPG ./downloads/*/*.*
rm ./downloads/*/*.jpg
