python n2.py 2
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python n2.py 3
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python n2.py 4
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python n2.py 5 
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python n2.py 6

