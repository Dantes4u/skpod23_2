#include <vector>   
#include <cmath>     
#include <iostream> 
#include <algorithm> 
#include <numeric>

class Grid {
private:
    std::vector<std::vector<std::vector<double> > > data;
    double dx, dy, dz;
    int N;

public:
    Grid(double Lx, double Ly, double Lz, int N) : dx(Lx / N), dy(Ly / N), dz(Lz / N), N(N) {
        data.resize(N + 1, std::vector<std::vector<double> >(
            N + 1, std::vector<double>(N + 1, 0.0)));
    }

    double& operator()(int x, int y, int z) {
        return data[x][y][z];
    }

    const double& operator()(int x, int y, int z) const {
        return data[x][y][z];
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

    void fillGrid(Grid& grid, double t) {
        double dx = grid.getDx();
        double dy = grid.getDy();
        double dz = grid.getDz();
        int N = grid.getN();
        for (int i = 0; i <= N; ++i) {
            for (int j = 0; j <= N; ++j) {
                for (int k = 0; k <= N; ++k) {
                    double x = i * dx;
                    double y = j * dy;
                    double z = k * dz;
                    grid(i, j, k) = equationFunction(x, y, z, t);
                }
            }
        }
    }

    void firstTimeStep(Grid& previous, Grid& current, double tau) {
        int N = previous.getN();
        for (int i = 0; i <= N; ++i) {
            for (int j = 1; j < N; ++j) {
                for (int k = 0; k <= N; ++k) {
                    current(i, j, k) = previous(i, j, k) + a2 * tau * tau / 2 * previous.laplacian(i, j, k);
                }
            }
        }
    }

    double equationFunction(double x, double y, double z, double t) {
        const double pi = acos(-1);
        double at = pi / 3.0 *sqrt((4.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 4.0 / (Lz * Lz)));
        return sin(2 * pi * x / Lx) * sin(pi * y / Ly + pi) * sin((2 * pi * z / Lz) + 2 * pi) * cos(at * t + pi);
    }

    void timeStep(Grid& current, Grid& previous, Grid& next, double tau) {
        int N = current.getN();
        for (int i = 0; i <= N; ++i) {
            for (int j = 1; j < N; ++j) {
                for (int k = 0; k <= N; ++k) {
                    next(i, j, k) = 2.0 * current(i, j, k) - previous(i, j, k) + tau * tau * a2 * current.laplacian(i, j, k);
                }
            }
        }
    }

    std::pair<double, double> calculateErrors(const Grid& numerical, double t) {
        int N = numerical.getN();
        double dx = numerical.getDx();
        double dy = numerical.getDy();
        double dz = numerical.getDz();
        
        std::vector<double> errors;
        errors.reserve((N+1) * (N+1) * (N+1)); 
        
        double max_error = 0.0;
        double error_sum = 0.0;
        double error = 0.0;

        for (int i = 0; i <= N; ++i) {
            for (int j = 0; j <= N; ++j) {
                for (int k = 0; k <= N; ++k) {
                    double x = i * dx;
                    double y = j * dy;
                    double z = k * dz;
                    double analytical_value = equationFunction(x, y, z, t);
                    double numerical_value = numerical(i, j, k);
                    error = std::abs(analytical_value - numerical_value);
                    errors.push_back(error);
                    max_error = std::max(max_error, error);
                }
            }
        }

        double average_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();

        return std::make_pair(average_error, max_error);
    }

    double getA2() const { return a2; }
};


int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <Lx> <Ly> <Lz> <N> <T> <n_steps>" << std::endl;
        return 1;
    }
    double Lx = atof(argv[1]);
    double Ly = atof(argv[2]);
    double Lz = atof(argv[3]);
    int N = atoi(argv[4]);
    double T = atof(argv[5]);
    int n_steps = atoi(argv[6]);
    Grid current(Lx, Ly, Lz, N);
    Grid next(Lx, Ly, Lz, N);
    Grid previous(Lx, Ly, Lz, N);
    

    Solver solver(Lx, Ly, Lz);
    double tau = T / n_steps;

    solver.fillGrid(previous, 0.0); 
    solver.firstTimeStep(previous, current, tau); 
    std::pair<double, double> errors = solver.calculateErrors(current, tau);
    double average_error = errors.first;
    double max_error = errors.second;
    std::cout<<1<<": "<<"average_error: "<<average_error<<"; max_error: "<<max_error<<std::endl;
    for (double t = 2*tau; t <= T; t += tau) {
        solver.timeStep(current, previous, next, tau);
        std::pair<double, double> errors = solver.calculateErrors(next, t);
        double average_error = errors.first;
        double max_error = errors.second;
        std::swap(previous, current); 
        std::swap(current, next);
        std::cout<<t/tau<<": "<<"average_error: "<<average_error<<"; max_error: "<<max_error<<std::endl;     
    }
    
    return 0;
}


