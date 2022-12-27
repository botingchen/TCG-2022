all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
train:
	./threes --total=200000 --block=1000 --limit=1000 --play="init save=weights.bin alpha=0.1 n_step=1"
load_train:
	./threes --total=20000 --block=1000 --limit=1000 --play="load=weights.bin save=weights.bin alpha=0.0005 n_step=3"
stats:
	./threes --total=1000 --save=stats.txt --play="load=weights.bin"
judge:
	./threes-judge --load stats.txt --judge version=2

#./threes --save=stats.txt
clean:
	rm threes
