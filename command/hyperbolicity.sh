# run script to produce hyperbolicity measure 

# document n (number of observations) and p (dimension) here
# tomato: 278, 25
# strawberry: 54, 98
# blueberry: 164, 50
# mnist: 70000, 662 
# cifar10: 50000, 1024

# setup parameters 
N=10            # repeat approximation 10 times
samples=1000    # number of sampels to measure 

# tomato 
for n in $(seq 1 $N); do 
    python src/measure_hyperbolicity.py --data tomato --num-samples $samples
    python src/measure_hyperbolicity.py --data syn --num-samples $samples --n 278 --p 25
done 

# strawberry 
for n in $(seq 1 $N); do 
    python src/measure_hyperbolicity.py --data strawberry --num-samples $samples
    python src/measure_hyperbolicity.py --data syn --num-samples $samples --n 54 --p 98
done 

# blueberry
for n in $(seq 1 $N); do 
    python src/measure_hyperbolicity.py --data blueberry --num-samples $samples
    python src/measure_hyperbolicity.py --data syn --num-samples $samples --n 164 --p 50
done 

# mnist
for n in $(seq 1 $N); do 
    python src/measure_hyperbolicity.py --data mnist --num-samples $samples
    python src/measure_hyperbolicity.py --data syn --num-samples $samples --n 70000 --p 662
done 

# cifar10
for n in $(seq 1 $N); do 
    python src/measure_hyperbolicity.py --data cifar10 --num-samples $samples
    python src/measure_hyperbolicity.py --data syn --num-samples $samples --n 50000 --p 1024
done 
