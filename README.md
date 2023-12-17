# skpod23
### Файлы
main.cpp - последовательная реализация

mympi.cpp - параллельная реализация

run.py - код для запуска друг за другом всех прогонов

videos - графики аналитической функции, численной функции и модуля ошибки

#### Компиляция mympi.cpp
`
module load SpectrumMPI

module load OpenMPI

mpixlC -o mpi mympi.cpp -qsmp=omp
`
#### Компиляция main.cpp
`
g++ main.cpp -o main
`
#### Запуск программы mpi (Lx, Ly, Lz, N, T, K)
`
mpisubmit.pl -p 2 -t 2 --stdout res.out --stderr tmp.err mpi -- 1.0 1.0 1.0 128 0.5 50
`
