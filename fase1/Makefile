CC = gcc
SRC = src/
CFLAGS = -O3 -ftree-vectorize -msse4 -mavx -pg 

.DEFAULT_GOAL = MD.exe

MD.exe: $(SRC)/MD.cpp
	$(CC) $(CFLAGS) $(SRC)MD.cpp -lm -o MD.exe

clean:
	rm ./MD.exe

run:
	srun --partition=cpar ./MD.exe < inputdata.txt

run1:
	srun --partition=cpar perf stat ./MD.exe < inputdata.txt

run2:
	perf stat ./MD.exe < inputdata.txt




