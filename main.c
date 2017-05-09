# include <math.h>
# include <mpi/mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>

double L = 1.0;            /* размер ячейки */
int N = 64;            /* число узлов в каждом измерении */

double *u, *u_new;        /* массивы для решения */

/* для более простого доступа к элементам матрицы (N+2)x(N+2) array */
#define INDEX(i, j) ((N+2)*(i)+(j))

int my_rank;            /* номер текущего процесса */

int *proc;            /* текущий узел */
int *i_min, *i_max;        /* минимальный и максимальный индексы узлов для процесса */
int *left_proc, *right_proc;    /* левый и правый соседи */

/*
  Предварительное объявление функций
*/
int main(int argc, char *argv[]);

void allocate_result_arrays();

void jacobi(int num_procs, double f[]);

void make_domains(int num_procs);

double *make_source();

void timestamp();

/******************************************************************************/

int main(int argc, char *argv[]) {
    double error;
    double max_error = 1.0E-3;
    double *f; /*сетка*/
    char *file_name = "poisson_mpi.out"; /*куда выводить результат*/
    double my_error;
    int my_count_of_points;
    int count_of_points;
    int num_procs; /*число процессов*/
    int step;
    double *swap;
    double wall_time;

    /*Инициализация*/
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /*Рапортуем об инициализации.*/
    if (my_rank == 0) {
        timestamp();
        printf("  count of processes         = %d\n", num_procs);
        printf("  count of vertices in a dimension = %d\n", N);
        printf("  max error = %f\n", max_error);
        printf("\n");
    }

    allocate_result_arrays();
    f = make_source();
    make_domains(num_procs);

    step = 0;

    /*Засекаем время.*/
    wall_time = MPI_Wtime();

    /*Начинаем итерации.*/
    do {
        jacobi(num_procs, f);
        ++step;

        /*Вычисляем ошибку.*/
        error = 0.0;
        count_of_points = 0;

        my_error = 0.0;
        my_count_of_points = 0;

        for (int i = i_min[my_rank]; i <= i_max[my_rank]; i++) {
            for (int j = 1; j <= N; j++) {
                if (u_new[INDEX(i, j)] != 0.0) {
                    my_error = my_error
                                + fabs(1.0 - u[INDEX(i, j)] / u_new[INDEX(i, j)]);

                    my_count_of_points = my_count_of_points + 1;
                }
            }
        }
        MPI_Allreduce(&my_error, &error, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        MPI_Allreduce(&my_count_of_points, &count_of_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (count_of_points != 0) {
            error = error / count_of_points;
        }
        if (my_rank == 0 && (step % 10) == 0) {
            printf("  N = %d, count_of_points = %d, my_count_of_points = %d, Step %4d  Error = %g\n",
                   N, count_of_points, my_count_of_points, step, error);
        }
        /*Обновляем решение.*/
        swap = u;
        u = u_new;
        u_new = swap;
    } while (max_error < error);

    /*Выводим решение.*/

    /*Выводим время.*/
    wall_time = MPI_Wtime() - wall_time;
    if (my_rank == 0) {
        printf("\n");
        printf(" time = %f secs\n", wall_time);
    }
    /*Заканчиваем и освобождаем ресурсы.*/
    MPI_Finalize();
    free(f);

    /*Рапортуем о завершении.*/
    if (my_rank == 0) {
        printf("\n");
        printf("  The end.\n");
        printf("\n");
        timestamp();
    }

    return 0;
}

/******************************************************************************/

void allocate_result_arrays()
{
    int ndof;

    ndof = (N + 2) * (N + 2);

    u = (double *) malloc(ndof * sizeof(double));
    for (int i = 0; i < ndof; i++) {
        u[i] = 0.0;
    }

    u_new = (double *) malloc(ndof * sizeof(double));
    for (int i = 0; i < ndof; i++) {
        u_new[i] = 0.0;
    }

    return;
}

/******************************************************************************/

void jacobi(int num_procs, double f[])
{
    double h = L / (double) (N + 1);;
    MPI_Request request[4];
    int requests;
    MPI_Status status[4];


    /*Обменялись данными с соседями.*/
    requests = 0;

    if (left_proc[my_rank] >= 0 && left_proc[my_rank] < num_procs) {
        MPI_Irecv(u + INDEX(i_min[my_rank] - 1, 1), N, MPI_DOUBLE,
                  left_proc[my_rank], 0, MPI_COMM_WORLD,
                  request + requests++);

        MPI_Isend(u + INDEX(i_min[my_rank], 1), N, MPI_DOUBLE,
                  left_proc[my_rank], 1, MPI_COMM_WORLD,
                  request + requests++);
    }

    if (right_proc[my_rank] >= 0 && right_proc[my_rank] < num_procs) {
        MPI_Irecv(u + INDEX(i_max[my_rank] + 1, 1), N, MPI_DOUBLE,
                  right_proc[my_rank], 1, MPI_COMM_WORLD,
                  request + requests++);

        MPI_Isend(u + INDEX(i_max[my_rank], 1), N, MPI_DOUBLE,
                  right_proc[my_rank], 0, MPI_COMM_WORLD,
                  request + requests++);
    }

    /*Обновили данные по своей области.*/
    for (int i = i_min[my_rank] + 1; i <= i_max[my_rank] - 1; i++) {
        for (int j = 1; j <= N; j++) {
            u_new[INDEX(i, j)] =
                    0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                            u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                            h * h * f[INDEX(i, j)]);
        }
    }

    /*завершили всё общение.*/
    MPI_Waitall(requests, request, status);

    /*Обновили решение у себя на границах.*/
    int i = i_min[my_rank];
    for (int j = 1; j <= N; j++) {
        u_new[INDEX(i, j)] =
                0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                        u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                        h * h * f[INDEX(i, j)]);
    }

    i = i_max[my_rank];
    if (i != i_min[my_rank]) {
        for (int j = 1; j <= N; j++) {
            u_new[INDEX(i, j)] =
                    0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                            u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                            h * h * f[INDEX(i, j)]);
        }
    }

    return;
}

/******************************************************************************/

void make_domains(int num_procs)
{
    double d;
    double eps;
    int i;
    int p;
    double x_max;
    double x_min;

    /*Создали массивы.*/
    proc = (int *) malloc((N + 2) * sizeof(int));
    i_min = (int *) malloc(num_procs * sizeof(int));
    i_max = (int *) malloc(num_procs * sizeof(int));
    left_proc = (int *) malloc(num_procs * sizeof(int));
    right_proc = (int *) malloc(num_procs * sizeof(int));

    /*Поделили зоны [(1-eps)..(N+eps)] между процессами.*/
    eps = 0.0001;
    d = (N - 1.0 + 2.0 * eps) / (double) num_procs;

    for (p = 0; p < num_procs; p++) {

        /*X_MIN <= I <= X_MAX.*/
        x_min = -eps + 1.0 + (double) (p * d);
        x_max = x_min + d;

        /* Для каждого узла с индексом I, запомнили PROC[I] процесс P, которому узел принадлежит.*/
        for (i = 1; i <= N; i++) {
            if (x_min <= i && i < x_max) {
                proc[i] = p;
            }
        }
    }

    /*Нашли минимальный индекс I для процесса P.*/
    for (p = 0; p < num_procs; p++) {
        for (i = 1; i <= N; i++) {
            if (proc[i] == p) {
                break;
            }
        }
        i_min[p] = i;

        /*Нашли максимальный индекс I для процесса P.*/
        for (i = N; 1 <= i; i--) {
            if (proc[i] == p) {
                break;
            }
        }
        i_max[p] = i;

        /*Нашли соседей слева и справа.*/
        left_proc[p] = -1;
        right_proc[p] = -1;

        if (proc[p] != -1) {
            if (1 < i_min[p] && i_min[p] <= N) {
                left_proc[p] = proc[i_min[p] - 1];
            }
            if (0 < i_max[p] && i_max[p] < N) {
                right_proc[p] = proc[i_max[p] + 1];
            }
        }
    }

    return;
}

/******************************************************************************/

double *make_source()
{
    double *f;
    int i;
    int j;
    int k;
    double q;

    f = (double *) malloc((N + 2) * (N + 2) * sizeof(double));

    for (i = 0; i < (N + 2) * (N + 2); i++) {
        f[i] = 0.0;
    }

    q = 10.0;

    i = 1 + N / 4;
    j = i;
    k = INDEX (i, j);
    f[k] = q;

    i = 1 + 3 * N / 4;
    j = i;
    k = INDEX (i, j);
    f[k] = -q;

    return f;
}

/******************************************************************************/

void timestamp()
{
# define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm *tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    printf("%s\n", time_buffer);

    return;
# undef TIME_SIZE
}
