all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
train:
	./threes --total=500000 --block=1000 --limit=1000 --play="init save=weights.bin alpha=0.1"
stats:
	./threes --total=1000 --save=stats.txt --play="load=weights.bin"
judge:
	./threes-judge --load stats.txt --judge version=2
	#./threes --save=stats.txt
clean:
	rm threes
