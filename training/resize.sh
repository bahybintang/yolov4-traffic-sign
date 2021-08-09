for i in *.jpg; do convert $i -resize 1024x -quality 50% $i; done
