//
// Created by adada on 9/1/2025.
//

#include "mc_read_load_compute.hpp"

/**
 * @brief Load data from pickle file into C++ array
 * @param filename Path to pickle file
 * @param data_ptr Pointer to array where data will be stored
 * @param size Expected size of data
 *
 * Process:
 * 1. Initialize Python interpreter and NumPy
 * 2. Open pickle file using Python's io module
 * 3. Load pickled NumPy array
 * 4. Convert to Python list
 * 5. Copy data to C++ array
 */
void mc_computation::load_pickle_data(const std::string &filename, double *data_ptr,
                                      std::size_t size)
{
    // Initialize Python interpreter (required for Boost.Python)
    Py_Initialize();

    // Initialize NumPy C API
    np::initialize();

    try
    {
        // Import Python's 'io' module for file operations
        py::object io_module = py::import("io");
        // Open file in binary read mode
        py::object file = io_module.attr("open")(filename, "rb");

        // Import pickle module for deserialization
        py::object pickle_module = py::import("pickle");

        // Deserialize the pickle file to Python object
        py::object loaded_data = pickle_module.attr("load")(file);

        // Close the file
        file.attr("close")();

        // Check if loaded object is a NumPy array
        if (py::extract<np::ndarray>(loaded_data).check())
        {
            // Extract as NumPy array
            np::ndarray np_array = py::extract<np::ndarray>(loaded_data);

            // Convert NumPy array to Python list for easier element access
            py::object py_list = np_array.attr("tolist")();

            // Get size of loaded data
            ssize_t list_size = py::len(py_list);

            // Verify size matches expected size
            if (static_cast<std::size_t>(list_size) > size)
            {
                throw std::runtime_error("The provided shared_ptr array size is smaller than the list size.");
            }

            // Copy data from Python list to C++ array
            for (ssize_t i = 0; i < list_size; ++i)
            {
                data_ptr[i] = py::extract<double>(py_list[i]);
            }
        }
        else
        {
            throw std::runtime_error("Loaded data is not a NumPy array.");
        }
    }
    catch (py::error_already_set &)
    {
        // Print Python error traceback
        PyErr_Print();
        throw std::runtime_error("Python error occurred.");
    }
}

/**
 * @brief Save C++ array to pickle file for Python compatibility
 * @param ptr Pointer to data array
 * @param size Number of elements in array
 * @param filename Output file path
 *
 * Process:
 * 1. Initialize Python/NumPy if needed
 * 2. Convert C++ array to NumPy array
 * 3. Serialize using pickle.dumps
 * 4. Write binary data to file
 */
void mc_computation::save_array_to_pickle(const double *ptr, int size, const std::string &filename)
{
    using namespace boost::python;
    namespace np = boost::python::numpy;

    // Initialize Python interpreter if not already initialized
    if (!Py_IsInitialized())
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Failed to initialize Python interpreter");
        }
        np::initialize(); // Initialize NumPy
    }

    try
    {
        // Import the pickle module
        object pickle = import("pickle");
        object pickle_dumps = pickle.attr("dumps");

        // Convert C++ array to NumPy array
        // np::from_data creates a NumPy array that views the C++ data
        np::ndarray numpy_array = np::from_data(
            ptr,                                       // Pointer to data
            np::dtype::get_builtin<double>(),          // Data type (double)
            boost::python::make_tuple(size),           // Shape of the array (1D array)
            boost::python::make_tuple(sizeof(double)), // Strides
            object()                                   // Optional base object
        );

        // Serialize the NumPy array using pickle.dumps
        object serialized_array = pickle_dumps(numpy_array);

        // Extract the serialized data as a string
        std::string serialized_str = extract<std::string>(serialized_array);

        // Write the serialized data to a file
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file for writing");
        }
        file.write(serialized_str.data(), serialized_str.size());
        file.close();

        // Debug output (optional)
        // std::cout << "Array serialized and written to file successfully." << std::endl;
    }
    catch (const error_already_set &)
    {
        PyErr_Print();
        std::cerr << "Boost.Python error occurred." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}

/**
 * @brief Initialize spin configuration from file or previous run
 *
 * If flushLastFile == -1:
 *   Load initial configuration from "s_init.pkl"
 * Else:
 *   Load final configuration from previous flush
 */
void mc_computation::init_s()
{
    std::string name;
    std::string s_inFileName;

    // Determine which file to load based on whether this is a new or continued simulation
    if (this->flushLastFile == -1)
    {
        // New simulation - load initial configuration
        name = "init";
        s_inFileName = this->out_s_path + "/s_" + name + ".pkl";
        this->load_pickle_data(s_inFileName, s_init, total_components_num);
    } // end flushLastFile==-1
    else
    {
        // Continue from previous simulation - load last saved configuration
        name = "flushEnd" + std::to_string(this->flushLastFile);
        s_inFileName = this->out_s_path + "/" + name + ".s_final.pkl";
        this->load_pickle_data(s_inFileName, s_init, total_components_num);

    } // end else
}

/**
 * @brief Convert 2D lattice coordinates to 1D flattened index
 * @param n0 Index in direction 0 (row)
 * @param n1 Index in direction 1 (column)
 * @return Flattened index = n0 * N1 + n1
 *
 * This is standard row-major ordering for 2D arrays
 */
long int mc_computation::ind(const int &n0, const int &n1)
{
    return 3 * (n0 * N1 + n1);
}

/**
 * @brief Main initialization and execution function
 *
 * Performs full simulation setup:
 * 1. Initialize spins from file
 * 2. Set up sublattices
 * 3. Construct neighbor patterns
 * 4. Initialize flattened neighbor lists
 * 5. Run Monte Carlo simulation
 */
void mc_computation::init_and_run()
{
    // Load initial or previous spin configuration
    this->init_s();

    // Execute Monte Carlo simulation
    this->execute_mc(s_init, newFlushNum);
}

/**
 * @brief Calculate total energy of configuration
 * @param s_vec Spin configuration
 * @return Total energy summing all interaction terms
 *
 * Computes:
 * - Sum of all Heisenberg nearest neighbor bonds
 * - Sum of all Heisenberg diagonal bonds
 * - Sum of all biquadratic nearest neighbor bonds
 * - Sum of all biquadratic diagonal bonds
 * - Sum of all Kitaev x bonds
 * - Sum of all Kitaev y bonds
 */
double mc_computation::energy_tot(const double *s_vec)
{
    double energy = 0.0;
    for (int x = 0; x < N0; x++)
    {
        for (int y = 0; y < N1; y++)
        {
            int xd = (x - 1 + N0) % N0;
            int xu = (x + 1) % N0;
            int yd = (y - 1 + N1) % N1;
            int yu = (y + 1) % N1;
            double dot_prod;
            double org_spin_x, org_spin_y, org_spin_z;

            org_spin_x = s_vec[ind(x, y)];
            org_spin_y = s_vec[ind(x, y) + 1];
            org_spin_z = s_vec[ind(x, y) + 2];

            // (x+1,y)
            long nn_site_ind = ind(xu, y);

            dot_prod = org_spin_x * s_vec[nn_site_ind] 
                        + org_spin_y * s_vec[nn_site_ind + 1] 
                        + org_spin_z * s_vec[nn_site_ind + 2];
            energy += (J11 + J12 * dot_prod) * dot_prod;   // nn Heisenberg S_i dot S_j and biquadratic term
            energy += K * org_spin_x * s_vec[nn_site_ind]; // Kitaev term Kx S_{i+e_x}^x S_i^x
            // (x,y-1)
            dot_prod = org_spin_x * s_vec[nn_site_ind] 
                       + org_spin_y * s_vec[nn_site_ind + 1] 
                       + org_spin_z * s_vec[nn_site_ind + 2];
            energy += (J11 + J12 * dot_prod) * dot_prod;       // nn Heisenberg S_i dot S_j and biquadratic term
            energy += K * org_spin_y * s_vec[nn_site_ind + 1]; // Kitaev term Ky S_{i+e_y}^y S_i^y

            // nnn interaction, Heisenberg, biquadratic, without Kitaev term
            // (x+1,y+1)

            nn_site_ind = ind(xu, yu);
            dot_prod = org_spin_x * s_vec[nn_site_ind] 
                       + org_spin_y * s_vec[nn_site_ind + 1] 
                       + org_spin_z * s_vec[nn_site_ind + 2];
            energy += (J21 + J22 * dot_prod) * dot_prod; // nnn Heisenberg S_i dot S_j and biquadratic term
            // (x-1,y+1)

            nn_site_ind = ind(xd, yu);
            dot_prod = org_spin_x * s_vec[nn_site_ind] 
                       + org_spin_y * s_vec[nn_site_ind + 1] 
                       + org_spin_z * s_vec[nn_site_ind + 2];
            energy += (J21 + J22 * dot_prod) * dot_prod; // nnn Heisenberg S_i dot S_j and biquadratic term
        }
    }
    return energy;
}

void mc_computation::local_update(unsigned int x, unsigned int y)
{

    int xd = (x - 1 + N0) % N0;
    int xu = (x + 1) % N0;
    int yd = (y - 1 + N1) % N1;
    int yu = (y + 1) % N1;

    // Thread-local random number generator and uniform distribution
    thread_local static std::mt19937 rng{std::random_device{}()};
    thread_local static std::uniform_real_distribution<double> dist{0.0, 1.0};
    // store original spin

    // new spin
    const double phi = 6.28318530718f * dist(rng);
    const double theta = std::acos(1.0 - 2.0 * dist(rng));
    // const double *new_spin = new double[3]{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
    double new_s_x = std::sin(theta) * std::cos(phi);
    double new_s_y = std::sin(theta) * std::sin(phi);
    double new_s_z = std::cos(theta);

    // calculate energy change
    double dE = 0;
    double old_dot, new_dot;

    long on_site_ind = ind(x, y);

    // nn interaction. Heisenberg, biquadratic and Kitaev terms
    // (x-1,y)

    long nn_site_ind = ind(xd, y);
    old_dot = s_init[on_site_ind] * s_init[nn_site_ind] + s_init[on_site_ind + 1] * s_init[nn_site_ind + 1] + s_init[on_site_ind + 2] * s_init[nn_site_ind + 2]; // S_i dot S_j
    new_dot = new_s_x * s_init[nn_site_ind] + new_s_y * s_init[nn_site_ind + 1] + new_s_z * s_init[nn_site_ind + 2];
    // dE += J11 * (new_dot - old_dot); // nn Heisenberg S_i dot S_j
    // dE += J12 * (new_dot * new_dot - old_dot * old_dot); // nn biquadratic term
    dE += (J11 + J12 * (new_dot + old_dot)) * (new_dot - old_dot);   // nn Heisenberg S_i dot S_j and biquadratic term
    dE += K * (new_s_x - s_init[on_site_ind]) * s_init[nn_site_ind]; // Kitaev term Kx S_{i+e_x}x ^ S_i^x

    // (x+1,y)
    nn_site_ind = ind(xu, y);
    old_dot = s_init[on_site_ind] * s_init[nn_site_ind] + s_init[on_site_ind + 1] * s_init[nn_site_ind + 1] + s_init[on_site_ind + 2] * s_init[nn_site_ind + 2]; // S_i dot S_j
    new_dot = new_s_x * s_init[nn_site_ind] + new_s_y * s_init[nn_site_ind + 1] + new_s_z * s_init[nn_site_ind + 2];
    // dE += J11 * (new_dot - old_dot); // nn Heisenberg S_i dot S_j
    // dE += J12 * (new_dot * new_dot - old_dot * old_dot); // nn biquadratic term
    dE += (J11 + J12 * (new_dot + old_dot)) * (new_dot - old_dot);   // nn Heisenberg S_i dot S_j and biquadratic term
    dE += K * (new_s_x - s_init[on_site_ind]) * s_init[nn_site_ind]; // Kitaev term Kx S_{i+e_x}x ^ S_i^x

    // (x,y-1)
    nn_site_ind = ind(x, yd);
    old_dot = s_init[on_site_ind] * s_init[nn_site_ind] + s_init[on_site_ind + 1] * s_init[nn_site_ind + 1] + s_init[on_site_ind + 2] * s_init[nn_site_ind + 2]; // S_i dot S_j
    new_dot = new_s_x * s_init[nn_site_ind] + new_s_y * s_init[nn_site_ind + 1] + new_s_z * s_init[nn_site_ind + 2];
    // dE += J11 * (new_dot - old_dot); // nn Heisenberg S_i dot S_j
    // dE += J12 * (new_dot * new_dot - old_dot * old_dot); // nn biquadratic term
    dE += (J11 + J12 * (new_dot + old_dot)) * (new_dot - old_dot);           // nn Heisenberg S_i dot S_j and biquadratic term
    dE += K * (new_s_x - s_init[on_site_ind + 1]) * s_init[nn_site_ind + 1]; // Kitaev term Kx S_{i+e_y}y ^ S_i^y

    // (x,y+1)
    nn_site_ind = ind(x, yu);
    old_dot = s_init[on_site_ind] * s_init[nn_site_ind] + s_init[on_site_ind + 1] * s_init[nn_site_ind + 1] + s_init[on_site_ind + 2] * s_init[nn_site_ind + 2]; // S_i dot S_j
    new_dot = new_s_x * s_init[nn_site_ind] + new_s_y * s_init[nn_site_ind + 1] + new_s_z * s_init[nn_site_ind + 2];
    // dE += J11 * (new_dot - old_dot); // nn Heisenberg S_i dot S_j
    // dE += J12 * (new_dot * new_dot - old_dot * old_dot); // nn biquadratic term
    dE += (J11 + J12 * (new_dot + old_dot)) * (new_dot - old_dot);           // nn Heisenberg S_i dot S_j and biquadratic term
    dE += K * (new_s_x - s_init[on_site_ind + 1]) * s_init[nn_site_ind + 1]; // Kitaev term Kx S_{i+e_y}y ^ S_i^y
    // nnn interaction, Heisenberg, biquadratic, without Kitaev term

    // (x-1,y-1)
    nn_site_ind = ind(xd, yd);
    old_dot = s_init[on_site_ind] * s_init[nn_site_ind] + s_init[on_site_ind + 1] * s_init[nn_site_ind + 1] + s_init[on_site_ind + 2] * s_init[nn_site_ind + 2]; // S_i dot S_j
    new_dot = new_s_x * s_init[nn_site_ind] + new_s_y * s_init[nn_site_ind + 1] + new_s_z * s_init[nn_site_ind + 2];
    dE += (J21 + J22 * (new_dot + old_dot)) * (new_dot - old_dot); // nnn Heisenberg S_i dot S_j and biquadratic term

    // (x+1,y+1)
    nn_site_ind = ind(xu, yu);
    old_dot = s_init[on_site_ind] * s_init[nn_site_ind] + s_init[on_site_ind + 1] * s_init[nn_site_ind + 1] + s_init[on_site_ind + 2] * s_init[nn_site_ind + 2]; // S_i dot S_j
    new_dot = new_s_x * s_init[nn_site_ind] + new_s_y * s_init[nn_site_ind + 1] + new_s_z * s_init[nn_site_ind + 2];
    dE += (J21 + J22 * (new_dot + old_dot)) * (new_dot - old_dot); // nnn Heisenberg S_i dot S_j and biquadratic term

    // (x-1,y+1)
    nn_site_ind = ind(xd, yu);
    old_dot = s_init[on_site_ind] * s_init[nn_site_ind] + s_init[on_site_ind + 1] * s_init[nn_site_ind + 1] + s_init[on_site_ind + 2] * s_init[nn_site_ind + 2]; // S_i dot S_j
    new_dot = new_s_x * s_init[nn_site_ind] + new_s_y * s_init[nn_site_ind + 1] + new_s_z * s_init[nn_site_ind + 2];
    dE += (J21 + J22 * (new_dot + old_dot)) * (new_dot - old_dot); // nnn Heisenberg S_i dot S_j and biquadratic term

    // (x+1,y-1)
    nn_site_ind = ind(xu, yd);
    old_dot = s_init[on_site_ind] * s_init[nn_site_ind] + s_init[on_site_ind + 1] * s_init[nn_site_ind + 1] + s_init[on_site_ind + 2] * s_init[nn_site_ind + 2]; // S_i dot S_j
    new_dot = new_s_x * s_init[nn_site_ind] + new_s_y * s_init[nn_site_ind + 1] + new_s_z * s_init[nn_site_ind + 2];
    dE += (J21 + J22 * (new_dot + old_dot)) * (new_dot - old_dot); // nnn Heisenberg S_i dot S_j and biquadratic term

    // Metropolis, accept with probability exp(-dE/T)
    if (dist(rng) < std::exp(-dE / T))
    {
        // accept the new spin
        s_init[on_site_ind] = new_s_x;
        s_init[on_site_ind + 1] = new_s_y;
        s_init[on_site_ind + 2] = new_s_z;
    }
}

void mc_computation::checkerboard_update(int dx, int dy)
{
#pragma omp parallel for collapse(2)
    for (int x = dx; x < N0; x += 2)
    {
        for (int y = dy; y < N1; y += 2)
        {
            this->local_update(x, y);
        }
    }
}

void mc_computation::sweep(double *s_init)
{
    this->checkerboard_update(0, 0);
    this->checkerboard_update(0, 1);
    this->checkerboard_update(1, 0);
    this->checkerboard_update(1, 1);
}

/**
 * @brief Execute full Monte Carlo simulation
 * @param s_init Initial spin configuration (modified in-place)
 * @param flushNum Number of data flushes to perform
 *
 * Main simulation loop:
 * For each flush:
 *   For each sweep:
 *     - Update all spins (one full sweep)
 *     - Every sweep_multiple sweeps, save configuration
 *   - Compute magnetizations for all saved configurations
 *   - Save energy, magnetization, and final spin configuration to disk
 *   - Print timing information
 */
void mc_computation::execute_mc(double *s_init, const int &flushNum)
{
    // Calculate starting flush number (continues from previous if applicable)
    int flushThisFileStart = this->flushLastFile + 1;

    // Main loop over flushes
    for (int fls = 0; fls < flushNum; fls++)
    {
        // Start timing this flush
        const auto tMCStart{std::chrono::steady_clock::now()};

        // Perform Monte Carlo sweeps
        for (int swp = 0; swp < sweepToWrite * sweep_multiple; swp++)
        {
            // Update all spins (one complete sweep)
            this->sweep(s_init);

            // Save configuration every sweep_multiple sweeps
            if (swp % sweep_multiple == 0)
            {
                int swp_out = swp / sweep_multiple;
                // Compute and save total energy
                double energy_tot = this->energy_tot(s_init);
                this->U_data_all_ptr[swp_out] = energy_tot;
                // Copy spin configuration to storage array
                std::memcpy(s_all_ptr + swp_out * total_components_num, s_init, total_components_num * sizeof(double));
            } // end save to array
        } // end sweep for
        // Calculate flush number for this file
        int flushEnd = flushThisFileStart + fls;
        std::string fileNameMiddle = "flushEnd" + std::to_string(flushEnd);

        // Save energy data to pickle file
        std::string out_U_PickleFileName = out_U_path + "/" + fileNameMiddle + ".U.pkl";
        this->save_array_to_pickle(U_data_all_ptr, sweepToWrite, out_U_PickleFileName);

        // Compute order parameter for all saved configurations
        this->compute_all_magnetizations_parallel();
        // Save order parameter data
        std::string out_M_PickleFileName = this->out_M_path + "/" + fileNameMiddle + ".M.pkl";
        // save M
        this->save_array_to_pickle(M_all_ptr, 3 * sweepToWrite, out_M_PickleFileName);

        // Compute order parameter for all saved configurations
        this->compute_all_order_parameters_parallel();
        // Save order parameter data
        std::string out_order_parameter_PickleFileName = this->out_order_parameter_path + "/" + fileNameMiddle + ".order_parameter.pkl";
        this->save_array_to_pickle(order_parameter_all_ptr, 3 * sweepToWrite, out_order_parameter_PickleFileName);

        // Save final configuration (for continuing simulation later)
        std::string out_s_final_PickleFileName = this->out_s_path + "/" + fileNameMiddle + ".s_final.pkl";
        this->save_array_to_pickle(s_init, total_components_num, out_s_final_PickleFileName);

        // Print timing information
        const auto tMCEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_secondsAll{tMCEnd - tMCStart};
        std::cout << "flush " + std::to_string(flushEnd) + ": "
                  << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
    } // end flush for loop
}

/**
 * @brief Compute average magnetization for one configuration
 * @param Mx Output: x-component of magnetization
 * @param My Output: y-component of magnetization
 * @param Mz Output: z-component of magnetization
 * @param startInd Starting index in s_all_ptr
 * @param length Number of spin components (3*N0*N1)
 *
 * Computes: M_α = (1/N) * Σ_i s_α^i  for α = x, y, z
 */
void mc_computation::compute_M_avg_over_sites(double &Mx, double &My, double &Mz, const int &startInd, const int &length)
{
    double sum_x = 0, sum_y = 0, sum_z = 0;

    // Sum x-components (at indices 0, 3, 6, ...)
    for (int j = startInd; j < startInd + length; j += 3)
    {
        sum_x += this->s_all_ptr[j];
        sum_y += this->s_all_ptr[j];
        sum_z += this->s_all_ptr[j];
    } // end for j
    Mx = sum_x / static_cast<double>(lattice_num);
    My = sum_y / static_cast<double>(lattice_num);
    Mz = sum_z / static_cast<double>(lattice_num);
}

/// by qyc
/// @param val_x x-component of order_parameter
/// @param val_y y-component of order_parameter
/// @param val_z z-component of order_parameter
/// @param startInd Starting index in s_all_ptr for this configuration
/// @param length Number of components (should be 3*N0*N1)
void mc_computation::compute_order_parameter(double &val_x, double &val_y, double &val_z, const int &startInd, const int &length)
{
    double sum_x = 0, sum_y = 0, sum_z = 0;

    for (int j = startInd; j < startInd + length; j += 3)
    {
        int spin_index = (j - startInd) / 3;
        int n0 = spin_index / N1;
        short int phase = (2 * (n0 % 2) - 1);

        sum_x += (this->s_all_ptr[j]) * phase;
        sum_y += (this->s_all_ptr[j + 1]) * phase;
        sum_z += (this->s_all_ptr[j + 2]) * phase;
    }
    val_x = sum_x / static_cast<double>(lattice_num);
    val_y = sum_y / static_cast<double>(lattice_num);
    val_z = sum_z / static_cast<double>(lattice_num);
}

// order parameter parallel
void mc_computation::compute_all_magnetizations_parallel()
{
    int num_threads = num_parallel;
    int config_size = total_components_num; // 3*N0*N1
    int num_configs = sweepToWrite;
    std::vector<std::thread> threads;

    // Calculate how many configurations each thread will process
    int configs_per_thread = num_configs / num_threads;
    int remainder = num_configs % num_threads;

    // Launch threads
    for (int t = 0; t < num_threads; ++t)
    {
        // Calculate range of configurations for this thread
        int start_config = t * configs_per_thread;
        int end_config = (t == num_threads - 1) ? start_config + configs_per_thread + remainder
                                                : start_config + configs_per_thread;

        // Each thread processes a range of configurations
        threads.emplace_back([this, start_config, end_config, config_size]()
                             {
            for (int config_idx = start_config; config_idx < end_config; ++config_idx) {
                double Mx, My, Mz;
                int startInd = config_idx * config_size;

                // Calculate magnetization for this configuration
                this->compute_M_avg_over_sites(Mx, My, Mz, startInd, config_size);

                // Store results (3 values per configuration: Mx, My, Mz)
                int M_idx = config_idx * 3;
                this->M_all_ptr[M_idx] = Mx;
                this->M_all_ptr[M_idx + 1] = My;
                this->M_all_ptr[M_idx + 2] = Mz;
            } });
    }

    // Wait for all threads to complete
    for (auto &thread : threads)
    {
        thread.join();
    }
}

/** by qyc
 * @brief Compute order parameter for all saved configurations in parallel
 *
 * For each configuration, computes order parameter:
 * val_α = (1/N) * Σ_i phase_i * s_α^i  for α = x, y, z
 * where phase_i = (-1)^(n0 mod 2)
 *
 * Stores val_x, val_y, val_z for each configuration in order_parameter_all_ptr
 */
void mc_computation::compute_all_order_parameters_parallel()
{
    int num_threads = num_parallel;
    int config_size = total_components_num; // 3*N0*N1
    int num_configs = sweepToWrite;
    std::vector<std::thread> threads;

    // Calculate how many configurations each thread will process
    int configs_per_thread = num_configs / num_threads;
    int remainder = num_configs % num_threads;

    // Launch threads
    for (int t = 0; t < num_threads; ++t)
    {
        // Calculate range of configurations for this thread
        int start_config = t * configs_per_thread;
        int end_config = (t == num_threads - 1) ? start_config + configs_per_thread + remainder
                                                : start_config + configs_per_thread;

        // Each thread processes a range of configurations
        threads.emplace_back([this, start_config, end_config, config_size]()
                             {
            for (int config_idx = start_config; config_idx < end_config; ++config_idx) {
                double val_x, val_y, val_z;
                int startInd = config_idx * config_size;

                // Calculate order parameter for this configuration
                this->compute_order_parameter(val_x, val_y, val_z, startInd, config_size);

                // Store results (3 values per configuration: val_x, val_y, val_z)
                int order_idx = config_idx * 3;
                this->order_parameter_all_ptr[order_idx] = val_x;
                this->order_parameter_all_ptr[order_idx + 1] = val_y;
                this->order_parameter_all_ptr[order_idx + 2] = val_z;
            } });
    }

    // Wait for all threads to complete
    for (auto &thread : threads)
    {
        thread.join();
    }
}