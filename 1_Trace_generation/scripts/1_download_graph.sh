
#!/bin/bash

mkdir -p "../../data"
mkdir -p "../../data/raw"
cd "../../data/raw"
# set default value for variable
#DEFAULT_VALUE="https://snap.stanford.edu/data/wiki-topcats.txt.gz"

DEFAULT_VALUE="https://snap.stanford.edu/data/amazon0302.txt.gz"
# check if argument is empty, then use the default value
LINK=${1:-$DEFAULT_VALUE}

filename=$(basename "$LINK")
echo "$filename"

wget $LINK

echo "Unzip raw graph data"
gunzip $filename

new_filename="${filename%.*}"
echo "$new_filename"

sed -i '1,4d' $new_filename

# example: 
#1: ./1_download_graph.sh
#2: ./1_download_graph.sh https://snap.stanford.edu/data/wiki-topcats.txt.gz