#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>

#define MAX_DIM 50
#define MIN_DIM 20
#define NUM_MATRICES 5

// Function to add two matrices
void addMatrices(int *A, int *B, int *C, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] = A[i * n + j] + B[i * n + j];
}

// Function to subtract two matrices
void subtractMatrices(int *A, int *B, int *C, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] = A[i * n + j] - B[i * n + j];
}

// Function to perform matrix multiplication using Strassen's method
void multiplyStrassen(int **matrices, int numMatrices, int n, int *optimalOrder, int blockSize, int rank, int size, MPI_Comm comm)
{

    // Divide matrices into submatrices
    int halfN = n / 2;

    int ***A = &matrices[optimalOrder[0]];
    int ***B = &matrices[optimalOrder[1]];

    int ***A11, ***A12, ***A21, ***A22, ***B11, ***B12, ***B21, ***B22;
    if (n <= blockSize)
    {
        // Base case: Standard matrix multiplication
        int *result = (int *)malloc(n * n * sizeof(int));

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i * n + j] = 0;
                for (int k = 0; k < n; k++)
                {
                    result[i * n + j] += (matrices[optimalOrder[0]][i * n + k]) * (matrices[optimalOrder[1]][k * n + j]);
                    printf("%d ", result[i * n + j]);
                }
                printf("\n");
            }
        }

        free(result);
    }
    else
    {

        if (rank == 0)
        {
            // Allocate memory for submatrices
            for (int i = 0; i < numMatrices; i++)
            {
                A11[i] = (int *)malloc(halfN * halfN * sizeof(int));
                A12[i] = (int *)malloc(halfN * halfN * sizeof(int));
                A21[i] = (int *)malloc(halfN * halfN * sizeof(int));
                A22[i] = (int *)malloc(halfN * halfN * sizeof(int));
                B11[i] = (int *)malloc(halfN * halfN * sizeof(int));
                B12[i] = (int *)malloc(halfN * halfN * sizeof(int));
                B21[i] = (int *)malloc(halfN * halfN * sizeof(int));
                B22[i] = (int *)malloc(halfN * halfN * sizeof(int));
                // Send and receive data between processes (blocking call)
                // Send submatrices A21, A22, B11, B21 to process 1
                MPI_Send(A11[i], halfN * halfN, MPI_INT, 1, 0, comm);
                MPI_Send(A12[i], halfN * halfN, MPI_INT, 1, 0, comm);
                MPI_Send(A21[i], halfN * halfN, MPI_INT, 1, 0, comm);
                MPI_Send(A22[i], halfN * halfN, MPI_INT, 1, 0, comm);
                MPI_Send(B11[i], halfN * halfN, MPI_INT, 1, 0, comm);
                MPI_Send(B12[i], halfN * halfN, MPI_INT, 1, 0, comm);
                MPI_Send(B21[i], halfN * halfN, MPI_INT, 1, 0, comm);
                MPI_Send(B22[i], halfN * halfN, MPI_INT, 1, 0, comm);

                // ... (similarly for other submatrices)
            }
        }
        else
        {
            // Receive submatrices A21, A22, B11, B21 from process 0
            MPI_Recv(A11[0], halfN * halfN, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(A12[0], halfN * halfN, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(A21[0], halfN * halfN, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(A22[0], halfN * halfN, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(B11[0], halfN * halfN, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(B12[0], halfN * halfN, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(B21[0], halfN * halfN, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(B22[0], halfN * halfN, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

            // ... (similarly for other submatrices)
        }
    }

    // Populate submatrices
    for (int i = 0; i < numMatrices; i++)
    {
        for (int j = 0; j < halfN; j++)
        {
            for (int k = 0; k < halfN; k++)
            {
                A11[i][j * halfN + k] = (*A)[i][j * n + k];
                A12[i][j * halfN + k] = (*A)[i][j * n + k + halfN];
                A21[i][j * halfN + k] = (*A)[i][(j + halfN) * n + k];
                A22[i][j * halfN + k] = (*A)[i][(j + halfN) * n + k + halfN];

                B11[i][j * halfN + k] = (*B)[i][j * n + k];
                B12[i][j * halfN + k] = (*B)[i][j * n + k + halfN];
                B21[i][j * halfN + k] = (*B)[i][(j + halfN) * n + k];
                B22[i][j * halfN + k] = (*B)[i][(j + halfN) * n + k + halfN];
            }
        }
    }

    // Calculate Strassen's products with blocking
    int *P1 = (int *)malloc(halfN * halfN * sizeof(int));
    int *P2 = (int *)malloc(halfN * halfN * sizeof(int));
    int *P3 = (int *)malloc(halfN * halfN * sizeof(int));
    int *P4 = (int *)malloc(halfN * halfN * sizeof(int));
    int *P5 = (int *)malloc(halfN * halfN * sizeof(int));
    int *P6 = (int *)malloc(halfN * halfN * sizeof(int));
    int *P7 = (int *)malloc(halfN * halfN * sizeof(int));

    // P1 = (A11 + A22) * (B11 + B22)
    addMatrices(A11, A22, P1, halfN);
    addMatrices(B11, B22, P2, halfN);
    multiplyStrassen(&P1, &P2, P1, halfN, blockSize, rank, size, comm);

    // P2 = (A21 + A22) * B11
    addMatrices(A21, A22, P2, halfN);
    multiplyStrassen(&P2, &B11, P2, halfN, blockSize, rank, size, comm);

    // P3 = A11 * (B12 - B22)
    subtractMatrices(B12, B22, P3, halfN);
    multiplyStrassen(&A11, &P3, P3, halfN, blockSize, rank, size, comm);

    // P4 = A22 * (B21 - B11)
    subtractMatrices(B21, B11, P4, halfN);
    multiplyStrassen(&A22, &P4, P4, halfN, blockSize, rank, size, comm);

    // P5 = (A11 + A12) * B22
    addMatrices(A11, A12, P5, halfN);
    multiplyStrassen(&P5, &B22, P5, halfN, blockSize, rank, size, comm);

    // P6 = (A12 - A22) * (B21 + B22)
    subtractMatrices(A12, A22, P6, halfN);
    addMatrices(B21, B22, P6, halfN);
    multiplyStrassen(&P6, &P6, P6, halfN, blockSize, rank, size, comm);

    // P7 = (A11 - A21) * (B11 + B12)
    subtractMatrices(A11, A21, P7, halfN);
    addMatrices(B11, B12, P7, halfN);
    multiplyStrassen(&P7, &P7, P7, halfN, blockSize, rank, size, comm);

    // Combine results into the result matrix
    // C11 = P5 + P4 - P2 + P6
    addMatrices(P5, P4, P1, halfN);
    subtractMatrices(P1, P2, P1, halfN);
    addMatrices(P1, P6, P1, halfN);

    // C12 = P1 + P2
    addMatrices(P1, P2, P2, halfN);

    // C21 = P3 + P4
    addMatrices(P3, P4, P3, halfN);

    // C22 = P5 + P1 - P3 - P7
    addMatrices(P5, P1, P4, halfN);
    subtractMatrices(P4, P3, P4, halfN);
    subtractMatrices(P4, P7, P4, halfN);

    // P1 = A11 * (B12 - B22)
    multiplyStrassen(A11, B12, P1, halfN, blockSize, rank, size, comm);

    // Free memory for submatrices
    for (int i = 0; i < numMatrices; i++)
    {
        free(A11[i]);
        free(A12[i]);
        free(A21[i]);
        free(A22[i]);
        free(B11[i]);
        free(B12[i]);
        free(B21[i]);
        free(B22[i]);
        free(P1[i]);
        free(P2[i]);
        free(P3[i]);
        free(P4[i]);
        free(P5[i]);
        free(P6[i]);
        free(P7[i]);
    }
}


// Function to generate random matrices
void InitializeMatrices(int ***matrices, int *matrixDimensions, int numMatrices, int rank)
{
    *matrices = (int **)malloc(numMatrices * sizeof(int *));
    for (int i = 0; i < numMatrices; i++)
    {
        (*matrices)[i] = (int *)malloc(matrixDimensions[2 * i] * matrixDimensions[2 * i + 1] * sizeof(int));
        for (int j = 0; j < matrixDimensions[2 * i]; j++)
        {
            for (int k = 0; k < matrixDimensions[2 * i + 1]; k++)
            {
                // You may adjust the range of random values based on your requirements
                (*matrices)[i][j * matrixDimensions[2 * i + 1] + k] = rand() % 10;
            }
        }
    }
}

// Function to print the total number of scalar multiplications
void optimalOrderbyScalerMultiplications(int *matrixDimensions, int numMatrices)
{
    int cost[numMatrices][numMatrices];

    // Initialize cost array with zeros
    for (int i = 0; i < numMatrices; i++)
    {
        for (int j = 0; j < numMatrices; j++)
        {
            cost[i][j] = 0;
        }
    }

    // Calculate cost for chain length 2 to numMatrices
    for (int chainLength = 2; chainLength <= numMatrices; chainLength++)
    {
        for (int i = 0; i <= numMatrices - chainLength; i++)
        {
            int j = i + chainLength - 1;
            cost[i][j] = INT_MAX;

            for (int k = i; k < j; k++)
            {
                int currentCost = cost[i][k] + cost[k + 1][j] +
                                  matrixDimensions[i * 2] * matrixDimensions[k * 2 + 1] * matrixDimensions[j * 2 + 1];

                if (currentCost < cost[i][j])
                {
                    cost[i][j] = currentCost;
                }
            }
        }
    }

    // Print the total number of scalar multiplications (in the top-right cell)
    printf("Total number of scalar multiplications required : %d\n", cost[0][numMatrices - 1]);
}

// Function to print matrices and their dimensions
void DisplayDimension_and_Matrices(int **matrices, int *matrixDimensions, int numMatrices)
{
    for (int i = 0; i < numMatrices; i++)
    {
        printf("Matrix %d (%d x %d):\n", i + 1, matrixDimensions[2 * i], matrixDimensions[2 * i + 1]);
        for (int j = 0; j < matrixDimensions[2 * i]; j++)
        {
            for (int k = 0; k < matrixDimensions[2 * i + 1]; k++)
            {
                printf("%d ", matrices[i][j * matrixDimensions[2 * i + 1] + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{

    // initilizations

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double startTime, endTime;
    int **matrices;
    int matrixDimensions[NUM_MATRICES][2];
    int optimalOrder[NUM_MATRICES];




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

    // Allocate memory for matrices and implement the generateRandomMatrices function
    InitializeMatrices(&matrices, matrixDimensions[0], NUM_MATRICES, rank);


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

    // Find the optimal order on rank 0
    if (rank == 0)
    {

        DisplayDimension_and_Matrices(matrices, matrixDimensions[0], NUM_MATRICES);
        startTime = MPI_Wtime();
        optimalOrderbyScalerMultiplications(matrixDimensions[0], NUM_MATRICES);
        endTime = MPI_Wtime();
        printf("\n\n ******* Time in seconds to find optimal order by scalar multiplications: %f seconds ******** \n\n", endTime - startTime);
    }

    // Broadcast the optimal order and measure the collective time
    MPI_Bcast(optimalOrder, NUM_MATRICES, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Multiply matrices using Strassen's method

    if (rank == 0)
    {
        startTime = MPI_Wtime();
        multiplyStrassen(matrices, NUM_MATRICES, matrixDimensions[1], optimalOrder, 2,rank,size,MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        printf("\n\n *******  Time in seconds for Strassen's matrix multiplication: %f seconds\n", endTime - startTime);
    }

    // Free allocated memory for matrices
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        free(matrices[i]);
    }
    free(matrices);

    MPI_Finalize();

    return 0;
}
