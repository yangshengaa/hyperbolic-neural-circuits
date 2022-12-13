# mnist 
python src/train_ae.py --model HAE --data mnist --num-hidden-layers 2 --hidden-dim 20 --epochs 1000 
python src/train_ae.py --model AE --data mnist --num-hidden-layers 2 --hidden-dim 20 --epochs 1000

# cifar10 
python src/train_ae.py --model HAE --data cifar10 --num-hidden-layers 2 --hidden-dim 20 --epochs 1000 
python src/train_ae.py --model AE --data cifar10 --num-hidden-layers 2 --hidden-dim 20 --epochs 1000
