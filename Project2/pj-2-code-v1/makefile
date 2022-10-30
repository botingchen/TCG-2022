all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
train:
	#./threes --total=500000 --block=1000 --limit=1000 --play="init save=weights.bin alpha=0.1"
	weights_size="65536,65536,65536,65536" # 4x6-tuple
	./threes --total=0 --slide="init=$weights_size save=weights.bin" # generate a clean network
	for i in {1..50}; \
	do \
		./threes --total=100000 --block=1000 --limit=1000 --slide="load=weights.bin save=weights.bin alpha=0.1" | tee -a train.log; \
		./threes --total=1000 --slide="load=weights.bin alpha=0" --save="stats.txt"; \
	done
stats:
	./threes --total=1000 --save=stats.txt --play="load=weights.bin"
judge:
	./threes-judge --load stats.txt --judge version=2
	#./threes --save=stats.txt
clean:
	rm threes
