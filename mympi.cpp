#include <vector>   
#include <cmath>     
#include <iostream> 
#include <algorithm> 
#include <numeric>
#include <mpi.h>
#include <omp.h>


class Grid {
private:
    double* data;
    double dx, dy, dz;
    int N;
    int size_block[3];

public:
    Grid(int* input_size_block, double Lx, double Ly, double Lz, int N) : dx(Lx / N), dy(Ly / N), dz(Lz / N), N(N) {
        for (int i = 0; i < 3; ++i) {
            size_block[i] = input_size_block[i];
        }
        data = new double[size_block[0] * size_block[1] * size_block[2]]();
        std::fill_n(data, size_block[0] * size_block[1] * size_block[2], 0.0);
    }

    double& operator()(int x, int y, int z) {
        return data[x * size_block[1] * size_block[2] + y * size_block[2] + z];
    }

    const double& operator()(int x, int y, int z) const {
        return data[x * size_block[1] * size_block[2] + y * size_block[2] + z];
    }
    
    double* get_pointer(int i, int j, int k) {
        return &(*this)(i, j, k);
    }

    double laplacian(int i, int j, int k) const {
        int im1 = (i == 0) ? (N- 1) : (i - 1);
        int ip1 = (i == N) ? 1 : i + 1;
        int km1 = (k == 0) ? (N- 1) : (k - 1);
        int kp1 = (k == N) ? 1 : k + 1;
        double diff_x = ((*this)(im1, j, k) - 2.0 * (*this)(i, j, k) + (*this)(ip1, j, k)) / (dx * dx);
        double diff_y = ((*this)(i, j - 1, k) - 2.0 * (*this)(i, j, k) + (*this)(i, j + 1, k)) / (dy * dy);
        double diff_z = ((*this)(i, j, km1) - 2.0 * (*this)(i, j, k) + (*this)(i, j, kp1)) / (dz * dz);
        return diff_x + diff_y + diff_z;
    }

    void swap(Grid& other) {
        std::swap(data, other.data);
    }

    double getDx() const { return dx; }
    double getDy() const { return dy; }
    double getDz() const { return dz; }
    int getN() const { return N; }
};

class Solver {
private:
    double Lx, Ly, Lz; 
    double a2;         
public:
    Solver(double Lx, double Ly, double Lz) : Lx(Lx), Ly(Ly), Lz(Lz), a2(1.0 / 9.0) {}
    void fill_block(Grid& grid, int* start, int* size, double t) {
        double dx = grid.getDx();
        double dy = grid.getDy();
        double dz = grid.getDz();
        int N = grid.getN();
        for (int i = 0; i < size[0]; ++i) {
            for (int j = 0; j < size[1]; ++j) {
                for (int k = 0; k < size[2]; ++k) {
                    double x = (start[0] + i - 1) * dx;
                    double y = (start[1] + j - 1) * dy;
                    double z = (start[2] + k - 1) * dz;
                    grid(i, j, k) = equationFunction(x, y, z, t);
                }
            }
        }
    }

    std::pair<double, double> firstTimeStep(Grid& previous, Grid& current, int* start, int* size_block, double tau) {
        int N = current.getN();
        double dx = current.getDx();
        double dy = current.getDy();
        double dz = current.getDz();
        double max_error = 0.0;
        double error_sum = 0.0;
        #pragma omp parallel for reduction(+:error_sum) reduction(max:max_error)
        for (int i = 1; i < size_block[0] - 1; ++i) {
            for (int j = 1; j < size_block[1] - 1; ++j) {
                for (int k = 1; k < size_block[2] - 1; ++k) {
                    current(i, j, k) = previous(i, j, k) + a2 * tau * tau / 2 * previous.laplacian(i, j, k);
                    double grid_i = (start[0] + i - 1) * dx;
                    double grid_j = (start[1] + j - 1) * dy;
                    double grid_k = (start[2] + k - 1) * dz;
                    double analytical_value = equationFunction(grid_i, grid_j, grid_k, tau);
                    double numerical_value = current(i, j, k);
                    double error = std::abs(analytical_value - numerical_value);
                    error_sum += error;
                    max_error = std::max(max_error, error);
                }
            }
        }
        double average_error = error_sum / ((size_block[0] - 2) * (size_block[1] - 2) * (size_block[2] - 2));
        return std::make_pair(average_error, max_error);
    }

    double equationFunction(double x, double y, double z, double t) {
        const double pi = acos(-1);
        double at = pi / 3.0 *sqrt((4.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 4.0 / (Lz * Lz)));
        return sin(2 * pi * x / Lx) * sin(pi * y / Ly + pi) * sin((2 * pi * z / Lz) + 2 * pi) * cos(at * t + pi);
    }

    std::pair<double, double> timeStep(
        Grid& current, Grid& previous, Grid& next, int prev, 
        int after, int* start, int* size_block, int* coord, int* nproc, 
        MPI_Request request, MPI_Comm cart, 
        MPI_Datatype xt, MPI_Datatype yt, MPI_Datatype zt,
        double tau, double t
        ) {
        int N = current.getN();
        double dx = current.getDx();
        double dy = current.getDy();
        double dz = current.getDz();
        int first_flag, last_flag;

        first_flag = (coord[0] == 0) ? 1 : 0;
        last_flag = (coord[0] == nproc[0] - 1) ? 1 : 0;
        MPI_Cart_shift(cart, 0, 1, &prev, &after);
        MPI_Isend(current.get_pointer(1 + first_flag, 1, 1), 1, xt, prev, 1, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
        MPI_Isend(current.get_pointer(size_block[0] - 2 - last_flag, 1, 1), 1, xt, after, 2, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
        first_flag = (coord[1] == 0) ? 1 : 0;
        last_flag = (coord[1] == nproc[1] - 1) ? 1 : 0;
        MPI_Cart_shift(cart, 1, 1, &prev, &after);
        if (!first_flag) {
            MPI_Isend(current.get_pointer(1, 1, 1), 1, yt, prev, 3, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        } 
        if (!last_flag) {
            MPI_Isend(current.get_pointer(1, size_block[1] - 2, 1), 1, yt, after, 4, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }
        first_flag = (coord[2] == 0) ? 1 : 0;
        last_flag = (coord[2] == nproc[2] - 1) ? 1 : 0;
        MPI_Cart_shift(cart, 2, 1, &prev, &after);
        MPI_Isend(current.get_pointer(1, 1, 1 + first_flag), 1, zt, prev, 5, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
        MPI_Isend(current.get_pointer(1, 1, size_block[2] - 2 - last_flag), 1, zt, after, 6, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);

        MPI_Cart_shift(cart, 0, 1, &prev, &after);
        MPI_Recv(current.get_pointer(0, 1, 1), 1, xt, prev, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(current.get_pointer(size_block[0] - 1, 1, 1), 1, xt, after, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        first_flag = (coord[1] == 0) ? 1 : 0;
        last_flag = (coord[1] == nproc[1] - 1) ? 1 : 0;
        MPI_Cart_shift(cart, 1, 1, &prev, &after);
        if (!first_flag) {
            MPI_Recv(current.get_pointer(1, 0, 1), 1, yt, prev, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (!last_flag) {
            MPI_Recv(current.get_pointer(1, size_block[1] - 1, 1), 1, yt, after, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Cart_shift(cart, 2, 1, &prev, &after);
        MPI_Recv(current.get_pointer(1, 1, 0), 1, zt, prev, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(current.get_pointer(1, 1, size_block[2] - 1), 1, zt, after, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        double max_error = 0.0;
        double error_sum = 0.0;
        #pragma omp parallel for reduction(+:error_sum) reduction(max:max_error)
        for (int i = 1; i < size_block[0] - 1; ++i) {
            for (int j = 1; j < size_block[1] - 1; ++j) {
                for (int k = 1; k < size_block[2] - 1; ++k) {
                    int grid_i = start[0] + i - 1;
                    int grid_j = start[1] + j - 1;
                    int grid_k = start[2] + k - 1;
                    if (grid_j == 0 || grid_j == N) {
                        next(i, j, k) = 0;
                    } else {
                        next(i, j, k) = 2.0 * current(i, j, k) - previous(i, j, k) + tau * tau * a2 * current.laplacian(i, j, k);
                    }
                    double analytical_value = equationFunction(grid_i * dx, grid_j * dy, grid_k * dz, t);
                    double numerical_value = next(i, j, k);
                    double error = std::abs(analytical_value - numerical_value);
                    error_sum += error;
                    max_error = std::max(max_error, error);
                }
            }
        }
        double average_error = error_sum / ((size_block[0] - 2) * (size_block[1] - 2) * (size_block[2] - 2));
        return std::make_pair(average_error, max_error);
    }
    double getA2() const { return a2; }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    int world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request request = MPI_REQUEST_NULL;

    int nproc[3] = {0, 0, 0};
    int roll[3] = {true, true, true};
    int coord[3];
    MPI_Dims_create(world, 3, nproc);
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, nproc, roll, false, &cart);
    MPI_Cart_coords(cart, rank, 3, coord);

    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <Lx> <Ly> <Lz> <N> <T> <n_steps>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    double Lx = atof(argv[1]);
    double Ly = atof(argv[2]);
    double Lz = atof(argv[3]);
    int N = atoi(argv[4]);
    double T = atof(argv[5]);
    int n_steps = atoi(argv[6]);

    int start_block[3];
    int end_block[3];
    int size_block[3];
    for (int i = 0; i < 3; ++i) {
        int block_size = (N + 1) / nproc[i];
        start_block[i] = block_size * coord[i];
        end_block[i] = block_size * (coord[i] + 1);
        if (coord[i] == nproc[i] - 1) {
            end_block[i] += (N + 1) % nproc[i];
        }
        size_block[i] = end_block[i] - start_block[i] + 2;
    }
    
    Grid current(size_block, Lx, Ly, Lz, N);
    Grid next(size_block, Lx, Ly, Lz, N);
    Grid previous(size_block, Lx, Ly, Lz, N);

    Solver solver(Lx, Ly, Lz);
    double tau = T / n_steps;

    solver.fill_block(previous, start_block, size_block, 0.0); 

    double start_time = MPI_Wtime();
    std::pair<double, double> errors = solver.firstTimeStep(previous, current, start_block, size_block, tau); 
    double average_error = errors.first;
    double max_error = errors.second;
    std::cout<<1<<": "<<"average_error: "<<average_error<<"; max_error: "<<max_error<<std::endl;

    int begin[3] = {0, 0, 0};
    int xs[3] = {1, size_block[1] - 2, size_block[2]- 2};
    int ys[3] = {size_block[0]- 2, 1, size_block[2]- 2};
    int zs[3] = {size_block[0]- 2, size_block[1]- 2, 1};
    int prev, after;
    MPI_Datatype xt, yt, zt;
    MPI_Type_create_subarray(3, size_block, xs, begin, MPI_ORDER_C, MPI_DOUBLE, &xt);
    MPI_Type_create_subarray(3, size_block, ys, begin, MPI_ORDER_C, MPI_DOUBLE, &yt);
    MPI_Type_create_subarray(3, size_block, zs, begin, MPI_ORDER_C, MPI_DOUBLE, &zt);
    MPI_Type_commit(&xt);
    MPI_Type_commit(&yt);
    MPI_Type_commit(&zt);
    
    for (double t = 2*tau; t <= T; t += tau) {
        std::pair<double, double> errors = solver.timeStep(current, previous, next, prev, after, start_block, size_block, coord, nproc, request, cart, xt, yt, zt, tau, t);
        double average_error = errors.first;
        double max_error = errors.second;
        previous.swap(current); 
        current.swap(next);
        std::cout<<t/tau<<": "<<"average_error: "<<average_error<<"; max_error: "<<max_error<<std::endl;     
    }
    double time = MPI_Wtime() - start_time;
    double max_time;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::cout << "Time:" << max_time << std::endl;
    MPI_Finalize();
    return 0;
}


