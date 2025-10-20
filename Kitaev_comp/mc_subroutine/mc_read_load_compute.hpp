//
// Created by adada on 9/1/2025.
//

#ifndef MC_READ_LOAD_COMPUTE_HPP
#define MC_READ_LOAD_COMPUTE_HPP
// Include necessary libraries for file system operations, Python integration, and numerical computations

#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <array>
#include <cfenv> // for floating-point exceptions
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <omp.h>
// Namespace aliases for convenience
namespace fs = boost::filesystem;
namespace py = boost::python;
namespace np = boost::python::numpy;
// Mathematical constant PI
constexpr double PI = M_PI;

/**
 * @class mc_computation
 * @brief Main class for Monte Carlo simulation of a spin system on a 2D lattice
 *
 * This class implements a Monte Carlo simulation for a spin system with:
 * - Heisenberg interactions (nearest neighbor and diagonal)
 * - Biquadratic interactions (nearest neighbor and diagonal)
 * - Kitaev interactions (directional x and y)
 */
class mc_computation
{
public:
    /**
     * @brief Constructor - reads parameters from file and initializes the simulation
     * @param cppInParamsFileName Path to input parameter file
     *
     * The parameter file should contain (one per line):
     * 1. Temperature (T)
     * 2. J11 - Heisenberg coupling, nearest neighbor
     * 3. J12 - Biquadratic coupling, nearest neighbor
     * 4. J21 - Heisenberg coupling, diagonal
     * 5. J22 - Biquadratic coupling, diagonal
     * 6. K - Kitaev coupling
     * 7. N - Lattice size (must be even, lattice will be N×N)
     * 8. sweepToWrite - writing data for each sweepToWrite sweeps
     * 10. flushLastFile - Index of last flush file (-1 for new simulation)
     * 11. TDirRoot - Root directory for temperature data
     * 12. U_s_dataDir - Directory for energy and spin data
     * 13. sweep_multiple - Number of sweeps between saved configurations
     * 14. num_parallel - Number of parallel threads to use
     */
    mc_computation(const std::string &cppInParamsFileName)
    {
        // Open the parameter file
        std::ifstream file(cppInParamsFileName);
        if (!file.is_open())
        {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0; // Tracks which parameter we're reading
        // Read parameters line by line
        while (std::getline(file, line))
        {
            // Skip empty lines
            if (line.empty())
            {
                continue;
            }
            std::istringstream iss(line);
            // Read Temperature (T)
            if (paramCounter == 0)
            {
                iss >> T;
                if (T <= 0)
                {
                    std::cerr << "T must be >0" << std::endl;
                    std::exit(1);
                } // end if
                std::cout << "T=" << T << std::endl;
                this->beta = 1.0 / T;
                std::cout << "beta=" << beta << std::endl;
                paramCounter++;
                continue;
            } // end T

            // Read J11 - Heisenberg coupling for nearest neighbors
            if (paramCounter == 1)
            {
                iss >> J11;
                std::cout << "J11=" << J11 << std::endl;
                paramCounter++;
                continue;
            } // end J11
            // Read J12 - Biquadratic coupling for nearest neighbors
            if (paramCounter == 2)
            {
                iss >> J12;
                std::cout << "J12=" << J12 << std::endl;
                paramCounter++;
                continue;
            } // end J12

            // Read J21 - Heisenberg coupling for diagonal neighbors
            if (paramCounter == 3)
            {
                iss >> J21;
                std::cout << "J21=" << J21 << std::endl;
                paramCounter++;
                continue;
            } // end J21

            // Read J22 - Biquadratic coupling for diagonal neighbors
            if (paramCounter == 4)
            {
                iss >> J22;
                std::cout << "J22=" << J22 << std::endl;
                paramCounter++;
                continue;
            } // end J22
            // Read K - Kitaev coupling strength
            if (paramCounter == 5)
            {
                iss >> K;
                std::cout << "K=" << K << std::endl;
                paramCounter++;
                continue;
            } // end K

            // Read N - Lattice dimension (creates N×N lattice)
            if (paramCounter == 6)
            {
                iss >> N0;
                N1 = N0;
                if (N0 <= 0)
                {
                    std::cerr << "N must be >0" << std::endl;
                    std::exit(1);
                }
                if (N0 % 2 != 0)
                {
                    std::cerr << "N must be even" << std::endl;
                    std::exit(1);
                }
                std::cout << "N0=N1=" << N0 << std::endl;
                // Calculate total number of lattice sites
                this->lattice_num = N0 * N1;
                // Total spin components (3 per site: sx, sy, sz)
                this->total_components_num = 3 * N0 * N1;
                std::cout << "total_components_num=" << total_components_num << std::endl;
                std::cout << "lattice_num=" << lattice_num << std::endl;
                paramCounter++;
                continue;
            } // end N

            // Read sweepToWrite - number of Monte Carlo sweeps before writing data
            if (paramCounter == 7)
            {
                iss >> sweepToWrite;
                if (sweepToWrite <= 0)
                {
                    std::cerr << "sweepToWrite must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "sweepToWrite=" << sweepToWrite << std::endl;
                paramCounter++;
                continue;
            } // end sweepToWrite

            // Read newFlushNum - number of times to flush data to disk
            if (paramCounter == 8)
            {
                iss >> newFlushNum;
                if (newFlushNum <= 0)
                {
                    std::cerr << "newFlushNum must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "newFlushNum=" << newFlushNum << std::endl;
                paramCounter++;
                continue;
            } // end newFlushNum

            // Read flushLastFile - index of last saved flush (-1 means start from scratch)
            if (paramCounter == 9)
            {
                iss >> flushLastFile;
                std::cout << "flushLastFile=" << flushLastFile << std::endl;
                paramCounter++;
                continue;
            } // end flushLastFile

            // Read TDirRoot - root directory for temperature-dependent data
            if (paramCounter == 10)
            {
                iss >> TDirRoot;
                std::cout << "TDirRoot=" << TDirRoot << std::endl;
                paramCounter++;
                continue;
            } // end TDirRoot

            // Read U_s_dataDir - directory for energy (U) and spin (s) data
            if (paramCounter == 11)
            {
                iss >> U_s_dataDir;
                std::cout << "U_s_dataDir=" << U_s_dataDir << std::endl;
                paramCounter++;
                continue;

            } // end U_s_dataDir

            // Read sweep_multiple - perform sweep_multiple*sweepToWrite sweeps between saved configurations
            if (paramCounter == 12)
            {
                iss >> sweep_multiple;
                if (sweep_multiple <= 0)
                {
                    std::cerr << "sweep_multiple must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout << "sweep_multiple=" << sweep_multiple << std::endl;
                paramCounter++;
                continue;
            } // end sweep_multiple

            // Read num_parallel - number of parallel threads for computation
            if (paramCounter == 13)
            {
                iss >> this->num_parallel;
                std::cout << "num_parallel=" << num_parallel << std::endl;
                paramCounter++;
                continue;
            } // end num_parallel

        } // end while - finished reading all parameters
        // Allocate memory for data storage
        try
        {
            // Energy values for each saved configuration
            this->U_data_all_ptr = new double[sweepToWrite];

            // Spin components for all configurations (sx, sy, sz for each site)
            this->s_all_ptr = new double[sweepToWrite * total_components_num];

            // Initial spin components
            this->s_init = new double[total_components_num];
            this->M_all_ptr = new double[sweepToWrite * 3];

            // order parameter values (val_x,val_y,val_z) for each configuration
            this->order_parameter_all_ptr = new double[sweepToWrite * 3];
        }
        catch (const std::bad_alloc &e)
        {
            std::cerr << "Memory allocation error: " << e.what() << std::endl;
            std::exit(2);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            std::exit(2);
        }

        // Create output directories if they don't exist
        this->out_U_path = this->U_s_dataDir + "/U/";
        if (!fs::is_directory(out_U_path) || !fs::exists(out_U_path))
        {
            fs::create_directories(out_U_path);
        }

        this->out_s_path = this->U_s_dataDir + "/s/";
        if (!fs::is_directory(out_s_path) || !fs::exists(out_s_path))
        {
            fs::create_directories(out_s_path);
        }

        this->out_M_path = this->U_s_dataDir + "/M/";
        if (!fs::is_directory(out_M_path) || !fs::exists(out_M_path))
        {
            fs::create_directories(out_M_path);
        }

        this->out_order_parameter_path = this->U_s_dataDir + "/out_order/";
        if (!fs::is_directory(out_order_parameter_path) || !fs::exists(out_order_parameter_path))
        {
            fs::create_directories(out_order_parameter_path);
        }

    } // end constructor

    // Destructor

    /**
     * @brief Destructor - frees allocated memory
     */
    ~mc_computation()
    {
        delete[] U_data_all_ptr; // Use delete[] for arrays!
        delete[] s_all_ptr;
        delete[] s_init;
        delete[] M_all_ptr;
        delete[] order_parameter_all_ptr;
    } // end Destructor
public:
    /**
     * @brief Initialize the simulation and run Monte Carlo
     *
     * This function:
     * 1. Initializes spin configuration
     * 2. Sets up sublattices for checkerboard updates
     * 3. Constructs neighbor lists
     * 4. Executes the Monte Carlo simulation
     */
    void init_and_run();

    /**
     * @brief Execute Monte Carlo simulation
     * @param s_init Initial spin components (sx, sy, sz for each site)
     * @param flushNum Number of data flushes to perform
     *
     * Performs Monte Carlo sweeps, updates spins, computes energies,
     * and periodically saves data to disk.
     */
    void execute_mc(double *s_init, const int &flushNum);
    
    /**
     * @brief Compute magnetization for all saved configurations in parallel
     *
     * For each configuration, computes average magnetization:
     * M = (1/N) * sum of all spins
     * Stores Mx, My, Mz for each configuration in M_all_ptr
     */
    void compute_all_magnetizations_parallel();
    /**
     * @brief Compute average magnetization over all sites for one configuration
     * @param Mx Output: x-component of magnetization
     * @param My Output: y-component of magnetization
     * @param Mz Output: z-component of magnetization
     * @param startInd Starting index in s_all_ptr for this configuration
     * @param length Number of components (should be 3*N0*N1)
     */
    void compute_M_avg_over_sites(double &Mx, double &My, double &Mz, const int &startInd, const int &length);

    /** by qyc
     * @brief Compute order parameter for all saved configurations in parallel
     *
     * For each configuration, computes order parameter:
     * val_α = (1/N) * Σ_i phase_i * s_α^i  for α = x, y, z
     * where phase_i = (-1)^(n0 mod 2)
     *
     * Stores val_x, val_y, val_z for each configuration in order_parameter_all_ptr
     */
    void compute_all_order_parameters_parallel();
    /// by qyc
    /// @param val_x x-component of order_parameter
    /// @param val_y y-component of order_parameter
    /// @param val_z z-component of order_parameter
    /// @param startInd Starting index in s_all_ptr for this configuration
    /// @param length Number of components (should be 3*N0*N1)
    void compute_order_parameter(double &val_x, double &val_y, double &val_z, const int &startInd, const int &length);

    /**
     * @brief Perform one Monte Carlo sweep updating all spins in parallel
     * @param s_curr Current spin configuration (all sx, sy, sz values)
     *
     * Updates spins in checkerboard pattern. This ordering ensures no two neighboring spins are updated simultaneously.
     */
    void sweep(double *s_curr);

    /**
     * @brief Calculate total energy of a configuration
     * @param s_vec Spin configuration (all sx, sy, sz values)
     * @return Total energy summing all interaction terms
     *
     * Computes:
     * E = 0.5 * (E_Heisenberg_nn + E_Heisenberg_diag +
     *            E_biquadratic_nn + E_biquadratic_diag +
     *            E_Kitaev_x + E_Kitaev_y)
     *
     * Factor of 0.5 corrects for double-counting of bonds
     */
    double energy_tot(const double *s_vec);

    /*
     * @brief Perform local update of spin at (x,y) by proposing new angles
     * @param x X-coordinate of spin to update
     * @param y Y-coordinate of spin to update
     */
    void local_update(unsigned int x, unsigned int y);

    void checkerboard_update(int dx, int dy);

    /**
     * @brief Extract spin components for a specific spin
     * @param s_vec Array containing all spin components
     * @param flattened_ind Index of spin
     * @param s_x Output: x-component
     * @param s_y Output: y-component
     * @param s_z Output: z-component
     */
    inline void get_spin_components(const double *s_vec, int flattened_ind,
                                    double &s_x, double &s_y, double &s_z)
    {
        // Each spin has 3 components, so multiply index by 3
        const double *spin_ptr = s_vec + (flattened_ind * 3);
        s_x = spin_ptr[0]; // First component is sx
        s_y = spin_ptr[1]; // Second component is sy
        s_z = spin_ptr[2]; // Third component is sz
    }

    /**
     * @brief Initialize spin configuration
     *
     * Either:
     * - Loads from saved file if continuing simulation (flushLastFile ≥ 0)
     * - Loads initial configuration if starting new simulation (flushLastFile = -1)
     */
    void init_s();

    /**
     * @brief Save array to pickle file for Python compatibility
     * @param ptr Pointer to data array
     * @param size Number of elements in array
     * @param filename Output file path
     *
     * Uses Boost.Python to serialize NumPy array to pickle format
     */
    void save_array_to_pickle(const double *ptr, int size, const std::string &filename);

    /**
     * @brief Load array from pickle file
     * @param filename Input file path
     * @param data_ptr Pointer to array where data will be stored
     * @param size Expected number of elements
     *
     * Uses Boost.Python to deserialize pickle file to C++ array
     */
    void load_pickle_data(const std::string &filename, double *data_ptr, std::size_t size);

    long int ind(const int &n0, const int &n1);

public:
    // ===== Physical Parameters =====
    double T;    ///< Temperature
    double beta; ///< Inverse temperature (1/T) for Boltzmann factor
    double J11;  ///< Heisenberg coupling, nearest neighbor
    double J12;  ///< Biquadratic coupling, nearest neighbor
    double J21;  ///< Heisenberg coupling, diagonal
    double J22;  ///< Biquadratic coupling, diagonal
    double K;    ///< Kitaev coupling strength

    // ===== Lattice Parameters =====
    int N0;                   ///< Lattice size in direction 0
    int N1;                   ///< Lattice size in direction 1 (equals N0 for square lattice)
    int lattice_num;          ///< Total number of sites (N0 × N1)
    int total_components_num; ///< Total spin components (3 × N0 × N1)

    // ===== Monte Carlo Parameters =====
    int sweepToWrite;   ///< Number of sweeps before writing data
    int newFlushNum;    ///< Number of data flushes
    int flushLastFile;  ///< Index of last saved flush (-1 for new simulation)
    int sweep_multiple; ///< Sweeps between saved configurations
    int num_parallel;   ///< Number of parallel threads

    // ===== Directory Paths =====
    std::string TDirRoot;                 ///< Root directory for temperature data
    std::string U_s_dataDir;              ///< Directory for energy and spin data
    std::string out_U_path;               ///< Output path for energy data
    std::string out_s_path;               ///< Output path for data
    std::string out_M_path;               ///< Output path for magnetization data
    std::string out_order_parameter_path; ///< Output path for order parameter data

    // ===== Data Storage =====
    double *U_data_all_ptr;          ///< Energy for each saved configuration
    double *s_all_ptr;               ///< Spin components for all configurations
    double *M_all_ptr;               ///< Magnetization (Mx, My, Mz) for all configurations
    double *s_init;                  ///< Initial spin configuration
    double *order_parameter_all_ptr; ///< order parameter (val_x, va_y, val_z) for all configurations
};

#endif // MC_READ_LOAD_COMPUTE_HPP