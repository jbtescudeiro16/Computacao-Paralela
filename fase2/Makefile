CC = gcc
SRC = src/
CFLAGS = -O3 -fopenmp -ftree-vectorize -msse4 -mavx  -march=native -funroll-loops -finline-functions -fomit-frame-pointer



.DEFAULT_GOAL = all

all: MDseq.exe MDpar.exe

MDseq.exe: $(SRC)/MDseq.cpp
	module load gcc/11.2.0;\
	$(CC) $(CFLAGS) $(SRC)MDseq.cpp -lm -o MDseq.exe

MDpar.exe: $(SRC)/MDpar.cpp
	module load gcc/11.2.0;\
	$(CC) $(CFLAGS) $(SRC)MDpar.cpp -lm -fopenmp -o MDpar.exe

clean:
	rm ./MD*.exe cp_output.txt cp_traj.xyz gmon.out cp_average.txt

runseq: MDseq.exe
	srun --partition=cpar ./MDseq.exe < inputdata.txt

runpar: MDpar.exe
	export OMP_NUM_THREADS=16;\
	./MDpar.exe < inputdata.txt

runparscript: MDpar.exe
	sbatch test.sh

runparperf: MDpar.exe
	export OMP_NUM_THREADS=16;\
	perf stat ./MDpar.exe < inputdata.txt
