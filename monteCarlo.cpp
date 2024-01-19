#include <iostream>
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>

using namespace std;

int main() {
    // File pointer
    fstream fin;

    // Open an existing file
    fin.open("/home/gbehrendt/CLionProjects/newSatellite/initialConditions.csv", ios::in);
    if (fin.is_open()) {
        cout << "File opened successfully :)" << endl;
    } else {
        cerr << "File not opened :(" << endl;
        return -1;
    }

    // Read the Data from the file
    // as String Vector
    std::vector<double> row;
    string item;
    int mcCount = 0;

    // Begin Monte Carlo loop
    while (getline(fin, item))
    {
        row.clear();
        istringstream line(item);
        while (getline(line, item, ','))
        {
            row.push_back(stod(item));
        }

        Eigen::MatrixXd storeRow = Eigen::Map<Eigen::Matrix<double, 14, 1> >(row.data());
        Eigen::MatrixXd x0(13, 1);
        for (int i = 0; i < 13; i++)
        {
            x0(i) = storeRow(i + 1);
        }
        cout << mcCount << " " << x0 << endl;
    }
    return 0;
}