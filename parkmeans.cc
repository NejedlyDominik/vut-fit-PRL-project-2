/**
 * PRL - Project 2 - Parallel K-means Clustering
 *
 * login: xnejed09
 * name: Dominik Nejedly
 * year: 2023
 *
 * Main module containing main function performing parallel 4-means Clustering algorithm
 */

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>

// predefined constants
#define SUCCESS 0
#define FAIL 1

#define ROOT_PROC 0
#define BYTE_SIZE 1
#define NUM_OF_CLUSTERS 4

#define INPUT_FILE "numbers"

/**
 * Compare two float values for equality.
 *
 * @param  num1     first float value
 * @param  num2     second float value
 * @param  epsilon  the maximum tolerance for two values to be considered equal (default value: 0.0001f)
 *
 * @return true if float values are equal otherwise false
 */
bool compareFloatsForEq(float num1, float num2, float epsilon=0.0001f) {
    return std::abs(num1 - num2) < epsilon;
}


/**
 * Get index of cluster for specified input value.
 *
 * @param  clusterMeans  mean cluster values
 * @param  num           value to be clustered
 *
 * @return index of the cluster to which the value belongs
 */
int getNumCluster(float *clusterMeans, uint8_t num) {
    int numCluster = 0;
    float centerDistance = std::abs(clusterMeans[0] - num);

    for(int i = 1; i < NUM_OF_CLUSTERS; i++) {
        float nextCenterDistance = std::abs(clusterMeans[i] - num);

        if(nextCenterDistance < centerDistance) {
            centerDistance = nextCenterDistance;
            numCluster = i;
        }
    }

    return numCluster;
}


/**
 * Print the clusters (their mean values) and their contents (values assigned to them).
 *
 * @param  resClustersOfNums  cluster indexes of all clustered values
 * @param  clusterMeans       mean cluster values
 * @param  numbers            all clustered values
 * @param  numLen             total number of clustered values
 */
void printClusters(int *resClustersOfNums, float *clusterMeans, std::vector<uint8_t> const &numbers, int numLen) {
    for(int i = 0; i < NUM_OF_CLUSTERS; i++) {
        bool firstVal = true;

        std::cout << "[" << clusterMeans[i] << "]";

        for(int j = 0; j < numLen; j++) {
            if(resClustersOfNums[j] == i) {
                if(firstVal) {
                    firstVal = false;
                }
                else {
                    std::cout << ",";
                }

                std::cout << " " << (unsigned) numbers.at(j);
            }
        }

        std::cout << "\n";
    }
}


int main(int argc, char **argv) {
    int rank, size;
    float clusterMeans[NUM_OF_CLUSTERS];
    std::vector<uint8_t> numbers;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Load numbers from input file and initialize cluster means.
    if(rank == ROOT_PROC) {
        std::ifstream inFile(INPUT_FILE, std::ios::binary);

        if(!inFile.is_open()) {
            std::cerr << "Input file " << INPUT_FILE << " cannot be opened.\n";
            MPI_Abort(MPI_COMM_WORLD, FAIL);
        }

        MPI_Comm_size(MPI_COMM_WORLD, &size);

        numbers.reserve(NUM_OF_CLUSTERS);
        const int minExpNumOfVals = (NUM_OF_CLUSTERS < size) ? size : NUM_OF_CLUSTERS;

        for(int i = 0; i < minExpNumOfVals; i++) {
            uint8_t inNum;

            if(inFile.read((char*) &inNum, BYTE_SIZE)) {
                numbers.push_back(inNum);
            }
            else {
                if(i < NUM_OF_CLUSTERS) {
                    std::cerr << "The input sequence contains less than " << NUM_OF_CLUSTERS << " values "
                        "(" << i << " - does not meet the minimum expected number of input values).\n";
                }
                else {
                    std::cerr << "The number of running processes (" << size << ") is greater than the number "
                        "of values in the input sequence (" << i << ").\n"; 
                }

                MPI_Abort(MPI_COMM_WORLD, FAIL);
            }
        }

        for(int i = 0; i < NUM_OF_CLUSTERS; i++) {
            clusterMeans[i] = (float) numbers.at(i);
        }
    }

    uint8_t num;

    // Scatter input values among all processes.
    MPI_Scatter(numbers.data(), 1, MPI_UINT8_T, &num, 1, MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);

    int numCluster, clusterMask[NUM_OF_CLUSTERS];
    int *clusterSums = NULL;
    int *clusterValCounts = NULL;

    if(rank == ROOT_PROC) {
        clusterSums = (int *) std::malloc(NUM_OF_CLUSTERS * sizeof(int));
        clusterValCounts = (int *) std::malloc(NUM_OF_CLUSTERS * sizeof(int));
    }

    bool doNextIter = true;

    // Loop until all cluster means of two consecutive iterations are identical.
    while(doNextIter) {
        // Broadcast current cluster means to all processes.
        MPI_Bcast(clusterMeans, NUM_OF_CLUSTERS, MPI_FLOAT, ROOT_PROC, MPI_COMM_WORLD);

        numCluster = getNumCluster(clusterMeans, num);
        std::memset(clusterMask, 0, sizeof(clusterMask));
        clusterMask[numCluster] = num;

        // Get the sum of values in each cluster.
        MPI_Reduce(clusterMask, clusterSums, NUM_OF_CLUSTERS, MPI_INT, MPI_SUM, ROOT_PROC, MPI_COMM_WORLD);

        clusterMask[numCluster] = 1;

        // Get the number of values in each cluster.
        MPI_Reduce(clusterMask, clusterValCounts, NUM_OF_CLUSTERS, MPI_INT, MPI_SUM, ROOT_PROC, MPI_COMM_WORLD);

        if(rank == ROOT_PROC) {
            doNextIter = false;

            // Compute new cluster means and decide whether to perform the next iteration.
            for(int i = 0; i < NUM_OF_CLUSTERS; i++) {
                if(clusterValCounts[i] > 0) {
                    float nextMean = (float) clusterSums[i] / clusterValCounts[i];

                    if(!compareFloatsForEq(clusterMeans[i], nextMean)) {
                        clusterMeans[i] = nextMean;
                        doNextIter = true;
                    }
                }
            }
        }

        // Broadcast the information about whether to perform the next iteration to all processes.
        MPI_Bcast(&doNextIter, 1, MPI_C_BOOL, ROOT_PROC, MPI_COMM_WORLD);
    }

    int *resClustersOfNums = NULL;

    if(rank == ROOT_PROC) {
        std::free(clusterSums);
        std::free(clusterValCounts);

        resClustersOfNums = (int *) std::malloc(size * sizeof(int));
    }

    // Gather cluster indexes of all clustered value.
    MPI_Gather(&numCluster, 1, MPI_INT, resClustersOfNums, 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);

    if(rank == ROOT_PROC) {
        printClusters(resClustersOfNums, clusterMeans, numbers, size);

        std::free(resClustersOfNums);
    }

    MPI_Finalize();
    return SUCCESS;
}
