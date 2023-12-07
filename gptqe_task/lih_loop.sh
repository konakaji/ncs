sleep 90m
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python lih.py 0 
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python lih.py 1
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python lih.py 2 
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python lih.py 3

