#include <cstdint>
#include <cstdlib>
#include <mpi.h>
#include <pthread.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>

#define MPI_ROOT_RANK 0
#define MAX_LEN 32
#define MIN_LEN 4
#define INPUT_FILE_NAME "numbers"
#define K 4

int main(int argc, char** argv) {

    MPI_Init(nullptr, nullptr);
    int process_cnt = -1;
    int current_process_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &process_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process_rank);

    int input_size = 0;
    int current_min_distance_idx = -1;
    double kmeans_centers[K] = {0.0};
    std::vector<uint8_t> program_input;

    // index represents the index of the number on the input and the value the index of the cluster that the number belongs to
    std::vector<int> cluster_indexes(process_cnt);
    // index represnets cluster and the value represents the sum of numbres assigned to the cluster
    std::vector<int> cluster_sums(K); 
    // index represents cluster and the value represents the count of numbres assignet to the cluster
    std::vector<int> cluster_cnts(K); 

    if (current_process_rank == MPI_ROOT_RANK) {
	FILE* input_handle = nullptr;
	
	input_handle = fopen("numbers","r");
	if (!input_handle) {
	    std::cerr << "[E]: Opening of file 'numbers' failed!" << std::endl;
	    return 1;
	}

	// read 8b numbers from the input file, but only the same amount as number of processes
	input_handle = fopen(INPUT_FILE_NAME,"r");
	int n, i;
	for (i = 0; (i < process_cnt && ((n = fgetc(input_handle)) != EOF)); i++)
	    program_input.push_back(n);
	fclose(input_handle);
	input_size = program_input.size();

	if (input_size < MIN_LEN || input_size > MAX_LEN) {
	    std::cerr << "[E]: Wrong input size!" << std::endl;
	    return 1;
	}

	if (input_size < process_cnt) {
	    std::cerr << "[E]: Wrong input - input size doesn't match number of processes!" << std::endl;
	    return 1;
	}

	// init centers - first K numbers from the input
	for (int i = 0; i < K; i++) {
	    kmeans_centers[i] = program_input[i];
	}
	
	// init root recv buffers
	std::fill(cluster_indexes.begin(), cluster_indexes.end(), 0);
	std::fill(cluster_cnts.begin(), cluster_cnts.end(), 0);
	std::fill(cluster_sums.begin(), cluster_sums.end(), 0);
    }
    
    // Distribute the program input, one number for eacho process
    uint8_t n_to_classify;
    MPI_Scatter(program_input.data(), 1, MPI_BYTE,
		&n_to_classify, 1, MPI_BYTE,
		MPI_ROOT_RANK, MPI_COMM_WORLD);
    
    bool end = false;
    while (!end) {

	// Distrubyte current centers to each process
	MPI_Bcast(kmeans_centers, K, MPI_DOUBLE, MPI_ROOT_RANK, MPI_COMM_WORLD);
	
	// Find the minimal distance from the centers
	std::vector<double> distances;
	for (int i = 0; i<K; i++) {
	    distances.push_back(abs(kmeans_centers[i] - n_to_classify));
	}
	std::vector<double>::iterator minimal_distance_it = std::min_element(distances.begin(), distances.end());
	current_min_distance_idx = std::distance(distances.begin(), minimal_distance_it); 
	
	// Sum of the index masks produces vector of amounts of numbers in each cluster
	std::vector<int> process_cluster_index(K);
	process_cluster_index[current_min_distance_idx] = 1;
	MPI_Reduce(process_cluster_index.data(), cluster_cnts.data(), K,
		   MPI_INT, MPI_SUM, MPI_ROOT_RANK, MPI_COMM_WORLD);
	
	// Similarly, sum of the value vectors produces vector of summed numbers in each cluster
	std::vector<int> process_cluster_value(K);
	process_cluster_value[current_min_distance_idx] = n_to_classify;
	MPI_Reduce(process_cluster_value.data(), cluster_sums.data(), K,
		   MPI_INT, MPI_SUM, MPI_ROOT_RANK, MPI_COMM_WORLD);
	
	// Calculate new centers and decide if the current state is the final and the while loop should terminate
	if (current_process_rank == MPI_ROOT_RANK) {
	    end = true;
	    for (int i = 0; i < K; i++) {
		double old_kmeans_center = kmeans_centers[i];
		if (cluster_cnts[i] != 0) {
		    kmeans_centers[i] = double(cluster_sums[i]) / double(cluster_cnts[i]);
		}
		end = end && (kmeans_centers[i] == old_kmeans_center);
	    }
	}

	MPI_Bcast(&end, 1, MPI_CXX_BOOL, MPI_ROOT_RANK, MPI_COMM_WORLD);
		
    }
    
    // Gather the cluster indexes for each number in the rank order
    MPI_Gather(&current_min_distance_idx, 1, MPI_INT,
	       cluster_indexes.data(), 1, MPI_INT, 
	       MPI_ROOT_RANK, MPI_COMM_WORLD);
    
    // Output the numbers for each cluster
    if (current_process_rank == MPI_ROOT_RANK) {
	for (int cluster = 0; cluster < K; cluster++) {
	    bool first_num = true;
	    std::cout << "[" << std::fixed << std::setprecision(3) << kmeans_centers[cluster] << "] ";
	    for (int n_idx = 0; n_idx < program_input.size(); n_idx++) {
		if (cluster_indexes[n_idx] == cluster){
		    if (!first_num) std::cout << ", ";
		    first_num = false;
		    std::cout << unsigned(program_input[n_idx]);
		}
	    }
	    std::cout << std::endl;
	}
    }

    MPI_Finalize();
    return 0;
}
