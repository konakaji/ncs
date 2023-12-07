<<<<<<< HEAD
#salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python beh2.py 0
#salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
=======
salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
>>>>>>> 02c723acfc8f5cc564a46f22ce092b73754351f5
python beh2.py 1
#salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python beh2.py 2
#salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python beh2.py 3 
#salloc --nodes 1 --qos interactive --time 01:30:00 --constraint gpu --gpus 4 --account=m4461 --no-shell
python beh2.py 4

