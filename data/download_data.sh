echo "download AllNLI"
wget https://sbert.net/datasets/AllNLI.tsv.gz

echo "download sohu-dd"
wget https://raw.githubusercontent.com/shibing624/text2vec/master/examples/data/SOHU/dd-test.jsonl -O ./sohu-dd.jsonl
echo "download sohu-dc"
wget https://raw.githubusercontent.com/shibing624/text2vec/master/examples/data/SOHU/dc-test.jsonl -O ./sohu-dc.jsonl