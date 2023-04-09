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


void print_byte_array(std::vector<uint8_t> array, std::string name) {
    for (int i = 0; i<array.size(); i++) {
	std::cout << unsigned(array[i]);
	if (array.size() != i+1) {
	    std::cout << ", ";
	} else {
	    std::cout << std::endl;
	}
    }
}


int main(int argc, char** argv) {

    MPI_Init(nullptr, nullptr);

    int process_cnt, current_process_rank;
    double kmeans_centers[K] = {0.0};
    process_cnt = current_process_rank = -1;
    std::vector<uint8_t> program_input;
    std::vector<int> cluster_indexes;
    int input_size = 0;

    std::vector<bool> are_centers_same_as_old(K);
    std::fill(are_centers_same_as_old.begin(), are_centers_same_as_old.end(), false);
    bool process_are_centers_same_as_old[K] = {false};
	
    MPI_Comm_size(MPI_COMM_WORLD, &process_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process_rank);

    if (current_process_rank == MPI_ROOT_RANK) {
	FILE* input_handle = nullptr;
	
	input_handle = fopen("numbers","r");
	if (!input_handle) {
	    std::cerr << "[E]: Opening of file 'numbers' failed!" << std::endl;
	    return 1;
	}

	//read from file	
	input_handle = fopen(INPUT_FILE_NAME,"r");
	int n, i;
	for (i = 0; (i < MAX_LEN && ((n = fgetc(input_handle)) != EOF)); i++)
	    program_input.push_back(n);
	fclose(input_handle);
	input_size = program_input.size();

	// TODO: remove DEBUG output
	//print_byte_array(program_input, "Program input: ");
	
	if (input_size < MIN_LEN || input_size > MAX_LEN) {
	    std::cerr << "[E]: Wrong input size!" << std::endl;
	    return 1;
	}

	if (input_size < process_cnt) {
	    std::cerr << "[E]: Wrong input - input size doesn't match number of processes!" << std::endl;
	    return 1;
	}

	// initialize centers
	for (int i = 0; i < K; i++) {
	    kmeans_centers[i] = program_input[i];
	}

        // initialize reduce point for the root
        // cluster_indexes is the place where are the indexes for each input
        // number gathered from the processes - maps the index of cluster to the
        // number from input based on the index in the vector since program_input
	// and cluster_indexes are the same length
        for (int i = 0; i < program_input.size(); i++) {
	    cluster_indexes.push_back(0);
	}
    }
    
    uint8_t n_to_cassify;
    MPI_Bcast(&input_size, 1, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Scatter(program_input.data(), 1, MPI_BYTE,
		&n_to_cassify, 1, MPI_BYTE,
		MPI_ROOT_RANK, MPI_COMM_WORLD);
    
    bool end = false;
    while (!end) {

	// TODO: REMOVE DEBUG OUTPUT
	/*
	std::cout << "Process" << current_process_rank << " centers_same_as_old: ";
	for (int i = 0; i < K; i++) {
	    std::cout << process_are_centers_same_as_old[i] << " ";
	}
	std::cout << std::endl;
	*/

	MPI_Bcast(kmeans_centers, K, MPI_DOUBLE, MPI_ROOT_RANK, MPI_COMM_WORLD);
	
	// TODO: REMOVE DEBUG OUTPUT
	/*
	std::cout << "Process " << current_process_rank << " centers: ";
	for (int i = 0; i < K; i++) {
	    std::cout << kmeans_centers[i]; 
	    if (i+1<K) { std::cout << ", "; } else { std::cout << std::endl; } 
	}
	std::cout << "Process " << current_process_rank << " to classify: " << unsigned(n_to_cassify) << std::endl;
	*/
    
	// Find the minimal distance from the centers
	std::vector<double> distances;
	for (int i = 0; i<K; i++) {
	    distances.push_back(abs(kmeans_centers[i] - n_to_cassify));
	}

	std::vector<double>::iterator minimal_distance_it = std::min_element(distances.begin(), distances.end());
	int minimal_distance_index = std::distance(distances.begin(), minimal_distance_it); 
    
	std::vector<int> process_cluster_indexes(input_size);
	std::fill(process_cluster_indexes.begin(), process_cluster_indexes.end(), 0);
	process_cluster_indexes[current_process_rank] = minimal_distance_index;
	
	MPI_Reduce(process_cluster_indexes.data(), cluster_indexes.data(), process_cluster_indexes.size(), 
		   MPI_INT, MPI_SUM, MPI_ROOT_RANK, MPI_COMM_WORLD);
	
	if (current_process_rank == MPI_ROOT_RANK) {
	    
	    // TODO: REMOVE DEBUG OUTPUT
	    /*
	    std::cout << "ROOT: Cluster indexes for the input are: ";
	    for (auto i : cluster_indexes) {
		std::cout << i << " ";
	    }
	    std::cout << std::endl;
	    */
	    
	    // Represents the total number of numbers assigned to each cluster
	    std::vector<int> cluster_contents_cnt(K);
	    std::fill(cluster_contents_cnt.begin(), cluster_contents_cnt.end(), 0);
	    
	    // Represents the sums of numbers assigned to each cluster
	    std::vector<int> cluster_contents_sums(K);
	    std::fill(cluster_contents_sums.begin(), cluster_contents_sums.end(), 0);

	    // Calculate the new centers
	    for (int i = 0; i < program_input.size(); i++) {
		cluster_contents_sums[cluster_indexes[i]] += program_input[i];
		cluster_contents_cnt[cluster_indexes[i]]++; 
	    }
	    
	    end = true;
	    for (int i = 0; i < K; i++) {
		double old_kmeans_center = kmeans_centers[i];
		if (cluster_contents_cnt[i] != 0) {
		    kmeans_centers[i] = double(cluster_contents_sums[i]) / double(cluster_contents_cnt[i]);
		}
		end = end && (kmeans_centers[i] == old_kmeans_center);
	    }
	}

	MPI_Bcast(&end, 1, MPI_CXX_BOOL, MPI_ROOT_RANK, MPI_COMM_WORLD);
    }

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
