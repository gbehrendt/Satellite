//
// Created by gbehrendt on 1/23/24.
//
#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
#pragma omp parallel for num_threads(4)
    for (int i = 1; i <= 10; i++) {
        int tid = omp_get_thread_num();
        printf("The thread %d  executes i = %d\n", tid, i);
    }
    cout << "Hello World!" << endl;

    return 0;
}