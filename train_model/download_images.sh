#!/bin/bash

rm -rf ./downloads/

for i in "art" "painting" "sculpture" "exhibition" "exhibit" "design" "kunstwerk";
do
	googleimagesdownload -k "$i" -f jpg -i art -l 500 -cd /usr/lib/chromium-browser/chromedriver
done

for i in "trash" "rubbish" "junk" "waste" "garbage" "scrap" "abfall";
do
	googleimagesdownload -k "$i" -f jpg -i trash -l 500 -cd /usr/lib/chromium-browser/chromedriver
done

mogrify -resize 224x224\! -colorspace sRGB -format JPG ./downloads/*/*.*
rm ./downloads/*/*.jpg
