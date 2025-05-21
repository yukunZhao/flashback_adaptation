rm src.tar.gz
wget http://10.223.11.146:8055/LLaMA-Factory/src.tar.gz ./
tar -xzvf src.tar.gz

target_path="/usr/local/lib/python3.8/dist-packages"
rm -rf $target_path/llamafactory
rm -rf $target_path/llamafactory.egg-info

mv src/* $target_path/

echo "done"
