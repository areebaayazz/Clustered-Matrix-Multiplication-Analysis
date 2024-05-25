#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#define MAX_DIM 50
#define MIN_DIM 20
#define NUM_MATRICES 5


// Function to calculate the scalar multiplications for matrix chain multiplication
int optimalOrderbyScalerMultiplications(int p[], int n, int m[][NUM_MATRICES], int s[][NUM_MATRICES])
{
    int i, j, k, l, q;

    for (i = 1; i < n; i++)
        m[i][i] = 0;

    for (l = 2; l < n; l++)
    {
        for (i = 1; i < n - l + 1; i++)
        {
            j = i + l - 1;
            m[i][j] = INT_MAX;
            for (k = i; k <= j - 1; k++)
            {
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (q < m[i][j])
                {
                    m[i][j] = q;
                    s[i][j] = k;
                }
            }
        }
    }

    return m[1][n - 1];
}



// Function to perform matrix multiplication using blocking
void MultiplyBlocking(int **matrices, int *matrixDimensions, int numMatrices, int rank, MPI_Comm comm)
{
    int i, j, k;
    int **resultMatrix;
    int *resultDimensions = matrixDimensions;
    MPI_Status status;

    // Allocate memory for the result matrix
    resultMatrix = (int **)malloc(resultDimensions[0] * sizeof(int *));
    for (i = 0; i < resultDimensions[0]; i++)
    {
        resultMatrix[i] = (int *)malloc(resultDimensions[1] * sizeof(int));
    }

    // Perform matrix multiplication using blocking
    for (i = 0; i < resultDimensions[0]; i++)
    {
        for (j = 0; j < resultDimensions[1]; j++)
        {
            resultMatrix[i][j] = 0;

            // MPI blocking communication (example using MPI_Send and MPI_Recv)
            // MPI_Send(&(matrices[0][i * matrixDimensions[1]]), matrixDimensions[1], MPI_INT, 1, 0, comm);
            // MPI_Recv(&(matrices[1][j]), 1, MPI_INT, 1, 0, comm, &status);

            // Perform local computation
            for (k = 0; k < matrixDimensions[1]; k++)
            {
                resultMatrix[i][j] += matrices[0][i * matrixDimensions[1] + k] * matrices[1][k * matrixDimensions[3] + j];
            }
        }
    }

    // Free memory for the result matrix
    for (i = 0; i < resultDimensions[0]; i++)
    {
        free(resultMatrix[i]);
    }
    free(resultMatrix);
}


// Function to perform matrix multiplication using blocking with non-blocking communication
void MultiplyNonBlocking(int **matrices, int *matrixDimensions, int numMatrices, int rank, MPI_Comm comm)
{
    int i, j, k;
    int **resultMatrix;
    int *resultDimensions = matrixDimensions;
    MPI_Status status;
    MPI_Request sendRequest, recvRequest;

    // Allocate memory for the result matrix
    resultMatrix = (int **)malloc(resultDimensions[0] * sizeof(int *));
    for (i = 0; i < resultDimensions[0]; i++)
    {
        resultMatrix[i] = (int *)malloc(resultDimensions[1] * sizeof(int));
    }

    // Perform matrix multiplication using blocking
    for (i = 0; i < resultDimensions[0]; i++)
    {
        for (j = 0; j < resultDimensions[1]; j++)
        {
            resultMatrix[i][j] = 0;

            // MPI non-blocking communication (example using MPI_Isend and MPI_Irecv)
            MPI_Isend(&(matrices[0][i * matrixDimensions[1]]), matrixDimensions[1], MPI_INT, 1, 0, comm, &sendRequest);
            MPI_Irecv(&(matrices[1][j]), 1, MPI_INT, 1, 0, comm, &recvRequest);

            // Perform local computation (note: This is still blocking locally)
            for (k = 0; k < matrixDimensions[1]; k++)
            {
                resultMatrix[i][j] += matrices[0][i * matrixDimensions[1] + k] * matrices[1][k * matrixDimensions[3] + j];
            }

            // MPI_Wait for non-blocking communication to complete (outside the inner loop)
            // MPI_Wait(&sendRequest, &status);
            // MPI_Wait(&recvRequest, &status);
        }
    }

    // Free memory for the result matrix
    for (i = 0; i < resultDimensions[0]; i++)
    {
        free(resultMatrix[i]);
    }
    free(resultMatrix);
}


// Function to print matrices and their dimensions
void DisplayMatrices(int **matrices, int *matrixDimensions, int numMatrices)
{
    for (int i = 0; i < numMatrices; i++)
    {
        printf("Matrix %d (%d x %d):\n", i + 1, matrixDimensions[2 * i], matrixDimensions[2 * i + 1]);
        for (int j = 0; j < matrixDimensions[2 * i]; j++)
        {
            for (int k = 0; k < matrixDimensions[2 * i + 1]; k++)
            {
                printf("%d\t", matrices[i][j * matrixDimensions[2 * i + 1] + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int **matrices;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime, endTime;

    int optimalOrder[NUM_MATRICES];
    int matrixDimensions[NUM_MATRICES][2];

    if (rank == 0)
    {
        // Read matrix dimensions from the file on rank 0
        FILE *file = fopen("dimensions.txt", "r");
        if (file == NULL)
        {
            perror("Error opening file");
            exit(1);
        }

        for (int i = 0; i < NUM_MATRICES; ++i)
        {
            if (fscanf(file, "%d X %d", &matrixDimensions[i][0], &matrixDimensions[i][1]) != 2)
            {
                fprintf(stderr, "Error reading matrix dimensions from file\n");
                exit(1);
            }
        }

        fclose(file);
    }

    // Broadcast matrix dimensions to all processes
    MPI_Bcast(matrixDimensions, NUM_MATRICES * 2, MPI_INT, 0, MPI_COMM_WORLD);

    // allocating space and initializing with random numbers
    matrices = (int **)malloc(NUM_MATRICES * sizeof(int *));
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        matrices[i] = (int *)malloc(matrixDimensions[i][0] * matrixDimensions[i][1] * sizeof(int));
    }

    // Generate random matrices directly in the main function
    srand(time(NULL) + rank); // Seed the random number generator with a different seed for each process
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        for (int j = 0; j < matrixDimensions[i][0]; j++)
        {
            for (int k = 0; k < matrixDimensions[i][1]; k++)
            {
                matrices[i][j * matrixDimensions[i][1] + k] = rand() % 10;
            }
        }
    }

    if (rank == 0)
    {
        // Broadcast each matrix to all processes
        for (int i = 0; i < NUM_MATRICES; ++i)
        {
            MPI_Bcast(&(matrices[i][0]), matrixDimensions[i][0] * matrixDimensions[i][1], MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Allocate memory for matrices on non-root processes
        matrices = (int **)malloc(NUM_MATRICES * sizeof(int *));
        for (int i = 0; i < NUM_MATRICES; ++i)
        {
            matrices[i] = (int *)malloc(matrixDimensions[i][0] * matrixDimensions[i][1] * sizeof(int));
        }
        // Broadcast each matrix to all processes
        for (int i = 0; i < NUM_MATRICES; ++i)
        {
            MPI_Bcast(&(matrices[i][0]), matrixDimensions[i][0] * matrixDimensions[i][1], MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0)
    {
        // Calculate scalar multiplications for matrix chain multiplication
        int m[NUM_MATRICES][NUM_MATRICES], s[NUM_MATRICES][NUM_MATRICES];

        int scalarMultiplications = optimalOrderbyScalerMultiplications((int *)matrixDimensions, NUM_MATRICES, m, s);
        
        // Print the matrices
        DisplayMatrices(matrices, (int *)matrixDimensions, NUM_MATRICES);

        // Print scalar multiplications
        printf("******  Number of Scalar Multiplications requiered for optimal order of multiplications *********: %d\n", scalarMultiplications);
    }

    // Record start time
    startTime = MPI_Wtime();
    // Perform matrix multiplication using blocking
    MultiplyBlocking(matrices, (int *)matrixDimensions, NUM_MATRICES, rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize processes

    if (rank == 0)
    {
        // Record end time
        endTime = MPI_Wtime();

        // Print the time taken
        printf("\n\n *******  Time in seconds for matrix multiplication by blocking call: %f seconds\n", endTime - startTime);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize processes

    if (rank == 0)
    {
        // Record start time
        startTime = MPI_Wtime();
    }

    // Perform matrix multiplication using non-blocking
    MultiplyNonBlocking(matrices, (int *)matrixDimensions, NUM_MATRICES, rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize processes

    if (rank == 0)
    {
        // Record end time
        endTime = MPI_Wtime();

        // Print the time taken
        printf("\n\n *******  Time in seconds for matrix multiplication by Non-blocking call: %f seconds\n", endTime - startTime);
    }

    MPI_Finalize();

    return 0;
}
