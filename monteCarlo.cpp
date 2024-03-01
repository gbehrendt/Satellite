#include <iostream>
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>

using namespace Eigen;
using namespace casadi;
using namespace std;
namespace fs = std::filesystem;

// Function Prototypes
MatrixXd Skew(VectorXd);
MatrixXd q2R(MatrixXd);
SX q2R(SX);
MatrixXd f(MatrixXd, MatrixXd, double, double);
SX f(SX, SX, SX, double, double);
void shiftRK4(int, double, MatrixXd &, MatrixXd,MatrixXd &, double, double);

int main() {
//    int maxIterArr[] = {10000};
    int maxIterArr[] = {5,6,7,8,9,10,15,50,100,125,150,175,200,300,400,500,1000,10000};
    int maxIterLength = sizeof(maxIterArr)/sizeof(maxIterArr[0]);
    int numConverged = 0;
    int mcCount = 0;
    std::vector<Function> solvers;

    for(int kk=0; kk < maxIterLength; kk++)
    {
        // File pointer
        fstream fin;
        // Open an existing file
        fin.open("/home/gbehrendt/CLionProjects/Satellite/initialConditions.csv", ios::in);
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
        mcCount = 0;
        numConverged = 0;

        // Initial Satellite Conditions + Problem Parameters
        double tPeriod = 92.68 * 60; // ISS orbital period (seconds)
        double n = -2 * M_PI / tPeriod; // Mean motion of ISS (rad/s)
        double mc = 12; // mass of the chaser

        // Begin Monte Carlo loop ***********************************************************************************
        while (getline(fin, item)) {
            row.clear();
            istringstream line(item);
            while (getline(line, item, ',')) {
                row.push_back(stod(item));
            }

            Eigen::MatrixXd storeRow = Eigen::Map<Eigen::Matrix<double, 14, 1> >(row.data());
            Eigen::MatrixXd x0(13, 1);
            for (int i = 0; i < 13; i++) {
                x0(i) = storeRow(i + 1);
            }
            cout << x0 << endl;

            // Declare states + controls
            SX u1 = SX::sym("u1"); // x thrust
            SX u2 = SX::sym("u2"); // y thrust
            SX u3 = SX::sym("u3"); // z thrust
            SX u4 = SX::sym("u4"); // x torque
            SX u5 = SX::sym("u5"); // y torque
            SX u6 = SX::sym("u6"); // z torque

            SX controls = vertcat(u1, u2, u3, u4, u5, u6);

            SX x = SX::sym("x");
            SX y = SX::sym("y");
            SX z = SX::sym("z");
            SX dx = SX::sym("dx");
            SX dy = SX::sym("dy");
            SX dz = SX::sym("dz");
            SX sq = SX::sym("sq");
            SX vq = SX::sym("vq", 3, 1);
            SX dw = SX::sym("dw", 3, 1);

            SX states = vertcat(x, y, z, dx, dy, dz);
            states = vertcat(states, sq, vq, dw);

            // Number of differential states
            const int numStates = states.size1();

            // Number of controls
            const int numControls = controls.size1();

            // Bounds and initial guess for the control
            double thrustMax = 1e-2;
            double thrustMin = -thrustMax;
            double torqueMax = 1e-4;
            double torqueMin = -torqueMax;
            std::vector<double> u_min = {thrustMin, thrustMin, thrustMin, torqueMin, torqueMin, torqueMin};
            std::vector<double> u_max = {thrustMax, thrustMax, thrustMax, torqueMax, torqueMax, torqueMax};
            std::vector<double> u_init = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            std::vector<double> x_min = {-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf};
            std::vector<double> x_max = {inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf};

            std::vector<double> xf_min = {-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf};
            std::vector<double> xf_max = {inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf};

            std::vector<double> x_init = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            double xPos = x0(0);
            double yPos = x0(1);
            double zPos = x0(2);
            double xVel = x0(3);
            double yVel = x0(4);
            double zVel = x0(5);
            std::vector<double> q0 = {x0(6), x0(7), x0(8), x0(9)};
            std::vector<double> dw0 = {x0(10), x0(11), x0(12)};

            // Bounds and initial guess for the state

            std::vector<double> x0_min = {xPos, yPos, zPos, xVel, yVel, zVel}; // initial position and velocity
            x0_min.insert(x0_min.end(), q0.begin(), q0.end()); // append initial quaternion
            x0_min.insert(x0_min.end(), dw0.begin(), dw0.end()); // append initial angular velocity

            std::vector<double> x0_max = {xPos, yPos, zPos, xVel, yVel, zVel};
            x0_max.insert(x0_max.end(), q0.begin(), q0.end()); // append initial quaternion
            x0_max.insert(x0_max.end(), dw0.begin(), dw0.end()); // append initial angular velocity


            // Tunable Parameters
            bool writeToFile = false; // choose to write to file or not
            string constraintType = "Euler"; // Choices: "RK4" or "Euler"
            const int N = 100; // Prediction Horizon
            double ts = 10.0; // sampling period
            int maxIter = maxIterArr[kk]; // maximum number of iterations IpOpt is allowed to compute per MPC Loop
            string hessianApprox = "exact";

            string timePath = "/home/gbehrendt/CLionProjects/Satellite/Timing/" + constraintType + "/ts" + to_string(to_int(ts)) + "/maxIter" + to_string(maxIter) + "/trial" + to_string(mcCount) + ".csv"; // insert your own file path
            string path = "/home/gbehrendt/CLionProjects/Satellite/Results/" + constraintType + "/ts" + to_string(to_int(ts)) + "/maxIter" + to_string(maxIter) + "/trial" + to_string(mcCount) + ".csv"; // insert your own file path


            double posCost = 1e5;
            double velCost = 1e2;
            double quatCost = 1e6;
            double angularCost = 1e7;
            double thrustCost = 1e5;
            double torqueCost = 1e10;

            // Total number of NLP variables
            const int numVars = numStates * (N + 1) + numControls * N;

            // Declare variable vector for the NLP

            SX V = SX::sym("V", numVars);

            // NLP variable bounds and initial guess
            std::vector<double> v_min, v_max, v_init;

            // Offset in V
            int offset = 0;

            // State at each shooting node and control for each shooting interval

            std::vector<SX> X, U;
            for (int k = 0; k < N; ++k) {
                // Local state
                X.push_back(V.nz(Slice(offset, offset + numStates)));
                if (k == 0) {
                    v_min.insert(v_min.end(), x0_min.begin(), x0_min.end());
                    v_max.insert(v_max.end(), x0_max.begin(), x0_max.end());
                } else {
                    v_min.insert(v_min.end(), x_min.begin(), x_min.end());
                    v_max.insert(v_max.end(), x_max.begin(), x_max.end());
                }
                v_init.insert(v_init.end(), x_init.begin(), x_init.end());
                offset += numStates;

                // Local control
                U.push_back(V.nz(Slice(offset, offset + numControls)));
                v_min.insert(v_min.end(), u_min.begin(), u_min.end());
                v_max.insert(v_max.end(), u_max.begin(), u_max.end());
                v_init.insert(v_init.end(), u_init.begin(), u_init.end());
                offset += numControls;
            }

            // State at end
            X.push_back(V.nz(Slice(offset, offset + numStates)));
            v_min.insert(v_min.end(), xf_min.begin(), xf_min.end());
            v_max.insert(v_max.end(), xf_max.begin(), xf_max.end());
            v_init.insert(v_init.end(), x_init.begin(), x_init.end());
            offset += numStates;

            // Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
            casadi_assert(offset == numVars, "");

            // Initialize Objective Function and Weighting Matrices
            SX J = 0; // Objective Function
            SX Q = SX::zeros(numStates, numStates);
            SX R = SX::zeros(numControls, numControls);

            Q(0, 0) = posCost; // xPos
            Q(1, 1) = posCost; // yPos
            Q(2, 2) = posCost; // zPos
            Q(3, 3) = velCost; // dx
            Q(4, 4) = velCost; // dy
            Q(5, 5) = velCost; // dz
            Q(6, 6) = quatCost; // sq
            Q(7, 7) = quatCost; // vq
            Q(8, 8) = quatCost; // vq
            Q(9, 9) = quatCost; // vq
            Q(10, 10) = angularCost; // dw
            Q(11, 11) = angularCost; // dw
            Q(12, 12) = angularCost; // dw

            R(0, 0) = thrustCost;
            R(1, 1) = thrustCost;
            R(2, 2) = thrustCost;
            R(3, 3) = torqueCost;
            R(4, 4) = torqueCost;
            R(5, 5) = torqueCost;

            SX xd = SX::zeros(numStates);
            xd(0) = 0.0;
            xd(1) = 0.0;
            xd(2) = 0.0;
            xd(3) = 0.0;
            xd(4) = 0.0;
            xd(5) = 0.0;
            xd(6) = 1.0;
            xd(7) = 0.0;
            xd(8) = 0.0;
            xd(9) = 0.0;
            xd(10) = 0.0;
            xd(11) = 0.0;
            xd(12) = 0.0;

            //Constraint function and bounds
            std::vector<SX> gAlgebraic;
            SX k1, k2, k3, k4, stNextRK4, stNextEuler;
            SX Rtc_k1, Rtc_k2, Rtc_k3, Rtc_k4;

            if (constraintType == "RK4") {
                // Loop over shooting nodes
                for (int k = 0; k < N; ++k) {
                    // algebraic definition of constraints
                    Rtc_k1 = q2R(X[k]);
                    k1 = f(X[k], U[k], Rtc_k1, n, mc);

                    Rtc_k2 = q2R(X[k] + (ts / 2) * k1);
                    k2 = f(X[k] + (ts / 2) * k1, U[k], Rtc_k2, n, mc);

                    Rtc_k3 = q2R(X[k] + (ts / 2) * k2);
                    k3 = f(X[k] + (ts / 2) * k2, U[k], Rtc_k3, n, mc);

                    Rtc_k4 = q2R(X[k] + ts * k3);
                    k4 = f(X[k] + ts * k3, U[k], Rtc_k4, n, mc);

                    stNextRK4 = X[k] + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4);

                    // Save continuity constraints
                    gAlgebraic.push_back(stNextRK4 - X[k + 1]);

                    // Add objective function contribution
                    J += mtimes(mtimes((X[k] - xd).T(), Q), (X[k] - xd)) + mtimes(mtimes(U[k].T(), R), U[k]);
                }
            } else if (constraintType == "Euler") {
                for (int k = 0; k < N; ++k) {
                    // algebraic definition of constraints
                    Rtc_k1 = q2R(X[k]);
                    stNextEuler = X[k] + ts * f(X[k], U[k], Rtc_k1, n, mc);

                    // Save continuity constraints
                    gAlgebraic.push_back(stNextEuler - X[k + 1]);

                    // Add objective function contribution
                    J += mtimes(mtimes((X[k] - xd).T(), Q), (X[k] - xd)) + mtimes(mtimes(U[k].T(), R), U[k]);
                }
            }

            // NLP
            SXDict nlp = {{"x", V},
                          {"f", J},
                          {"g", vertcat(gAlgebraic)}};

            // Set options
            Dict opts;
            if (writeToFile == true) {
                opts["ipopt.tol"] = 1e-5;
                opts["ipopt.max_iter"] = maxIter;
                opts["ipopt.hessian_approximation"] = hessianApprox; // for no max iterations change from "limited-memory" to "exact"
                opts["ipopt.print_level"] = 5;
                opts["ipopt.acceptable_tol"] = 1e-8;
                opts["ipopt.acceptable_obj_change_tol"] = 1e-6;
                opts["expand"] = false;
                opts["ipopt.file_print_level"] = 3;
                opts["ipopt.output_file"] = timePath;
                opts["ipopt.print_timing_statistics"] = "yes";
            } else if (writeToFile == false) {
                opts["ipopt.tol"] = 1e-5;
                opts["ipopt.max_iter"] = maxIter;
                opts["ipopt.hessian_approximation"] = hessianApprox; // for no max iterations change from "limited-memory" to "exact"
                opts["ipopt.print_level"] = 5;
                opts["ipopt.acceptable_tol"] = 1e-8;
                opts["ipopt.acceptable_obj_change_tol"] = 1e-6;
                opts["expand"] = false;
            }

            // Create an NLP solver and buffers
            std::map<std::string, DM> arg, res, sol;
            solvers.push_back(nlpsol("nlpsol", "ipopt", nlp, opts));

            // Bounds and initial guess
            arg["lbx"] = v_min;
            arg["ubx"] = v_max;
            arg["lbg"] = 0;
            arg["ubg"] = 0;
            arg["x0"] = v_init;

            //---------------------//
            //      MPC Loop       //
            //---------------------//
            Eigen::MatrixXd Storex0(numStates, 1);
            Storex0 = x0;

            // Define docking configuration
            Eigen::MatrixXd xs(numStates, 1);
            xs << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;


            Eigen::MatrixXd xx(numStates, N + 1);
            xx.col(0) = x0;
            Eigen::MatrixXd xx1(numStates, N + 1);
            Eigen::MatrixXd X0(numStates, N + 1);
            Eigen::MatrixXd u0;
            Eigen::MatrixXd uwu(numControls, N);
            Eigen::MatrixXd u_cl(numControls, N);

            vector<vector<double> > MPCstates(numStates);
            vector<vector<double> > MPCcontrols(numControls);


            for (int j = 0; j < numStates; j++) {
                MPCstates[j].push_back(x0(j));
            }

            // Start MPC
            int iter = 0;
            double epsilon = 1e-3;
            cout << numVars << endl;
            double infNormSt = 10;
            double infNormCon = 10;
            double infNorm = 100;

            while ( infNorm > epsilon && iter < N && infNorm < 1000) {
                // Solve NLP
                sol = solvers[0](arg);

                std::vector<double> V_opt(sol.at("x"));

                // Store Solution
                for (int i = 0; i <= N; ++i) {
                    for (int j = 0; j < numStates; ++j) {
                        xx1(j, i) = V_opt[j + i * (numStates + numControls)];
                    }
                    if (i < N) {
                        for (int j = 0; j < numControls; ++j) {
                            uwu(j, i) = V_opt[numStates + j + i * (numStates + numControls)];
                        }
                    }
                }
//                cout << "NLP States:" << endl << xx1 << endl;
//                cout <<endl;
//                cout << "NLP Controls:" << endl <<  uwu << endl;
//                cout <<endl;

                // Get solution Trajectory
                u_cl.col(iter) = uwu.col(0); // Store first control action from optimal sequence

                // Apply control and shift solution
                shiftRK4(N, ts, x0, uwu, u0, n, mc);
                xx(Eigen::placeholders::all, iter + 1) = x0;

                // Shift trajectory to initialize next step
                std::vector<int> ind(N); // vector with N-1 integers to be filled
                std::iota(std::begin(ind), std::end(ind), 1); // fill vector with N integers starting at 1
                X0 = xx1(Eigen::placeholders::all, ind); // assign X0 with columns 1-(N) of xx1
                X0.conservativeResize(X0.rows(), X0.cols() + 1);
                X0.col(X0.cols() - 1) = xx1(Eigen::placeholders::all, Eigen::placeholders::last);

//                cout << "MPC States:" << endl << xx << endl;
//                cout <<endl;
//                cout << "MPC Controls:" << endl << u_cl << endl << endl;

                for (int j = 0; j < numStates; j++) {
                    MPCstates[j].push_back(x0(j));
                }

                for (int j = 0; j < numControls; j++) {
                    MPCcontrols[j].push_back(u_cl(j, iter));
                }

                // Re-initialize Problem Parameters
                v_min.erase(v_min.begin(), v_min.begin() + numStates);
                for(int j = numStates-1; j>=0; j--)
                {
                    v_min.insert(v_min.begin(), x0(j));
                }

                v_max.erase(v_max.begin(), v_max.begin() + numStates);
                for(int j = numStates-1; j>=0; j--)
                {
                    v_max.insert(v_max.begin(), x0(j));
                }

                // Re-initialize initial guess
                v_init = V_opt;
                v_init.erase(v_init.begin(), v_init.begin() + (numStates + numControls));
                std::vector<double> finalStates;
                copy(v_init.end() - (numStates + numControls), v_init.end(), back_inserter(finalStates));
                v_init.insert(v_init.end(), finalStates.begin(), finalStates.end());


                arg.clear();
                arg["lbx"] = v_min;
                arg["ubx"] = v_max;
                arg["lbg"] = 0;
                arg["ubg"] = 0;
                arg["x0"] = v_init;


                infNormSt = (x0 - xs).lpNorm<Eigen::Infinity>();
                infNormCon = u_cl.col(iter).lpNorm<Eigen::Infinity>();
                infNorm = max(infNormSt, infNormCon); // l-infinity norm of current state and control
                cout << infNorm << endl;

                iter++;
                V_opt.clear();
                finalStates.clear();
                sol.clear();
                ind.clear();
                cout << iter << endl;
                cout << "Trial #" << mcCount << endl;
            }


            string converged;

            if (infNorm < epsilon) {
                converged = "yes";
                numConverged += 1;
                printf("*************************** maxIter: %d  Trial #%d = SUCCESS!!! *************************** \n",maxIterArr[kk],mcCount);
            }
            else {
                cout << "Trial #" << mcCount << " completed." << endl;
                converged = "no";
            }
            if (writeToFile == true) {
                const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
                ofstream fout; // declare fout variable

                fout.open(path, std::ofstream::out | std::ofstream::trunc); // open file to write to

                fout
                        << "x (km),y (km),z (km),xdot (km/s),ydot (km/s),zdot (km/s),sq,v1,v2,v3,dw1 (rad/s),dw2 (rad/s),dw3 (rad/s),thrust1 (N),thrust2 (N),thrust3 (N),tau1 (rad/s^2),tau2 (rad/s^2),tau3 (rad/s^2),x0,Maximum Iterations,ts,N,MPC Loops,posCost,velCost,quatCost,angualarCost,thrustCost,torqueCost,"
                           "thrustMax,torqueMax,Constraint Type,Trial,converged?,infNorm"
                        << endl;

                for (int j = 0; j < iter; j++) {
                    if (j == 0) {
                        for (int i = 0; i < (MPCstates.size() + MPCcontrols.size()); i++) {
                            if (i < numStates) {
                                fout << MPCstates[i][j] << ",";
                            } else if (i < numStates + numControls) {
                                fout << MPCcontrols[i - numStates][j] << ",";
                            }
                        }
                        fout << Storex0(j) << "," << maxIter << "," << ts << "," << N << "," << iter << "," << posCost
                             << "," << velCost << "," << quatCost << "," << angularCost << "," << thrustCost << "," << torqueCost << ","
                             <<  thrustMax << "," << torqueMax << "," << constraintType << "," << mcCount << ","
                             << converged << "," << infNorm;
                    } else {
                        for (int i = 0; i < (MPCstates.size() + MPCcontrols.size()); i++) {
                            if (i < numStates) {
                                fout << MPCstates[i][j] << ",";
                            } else if (i < numStates + numControls) {
                                fout << MPCcontrols[i - numStates][j] << ",";
                            }
                        }
                        if (j < numStates) {
                            fout << Storex0(j) << ",";
                        }
                    }
                    fout << endl;
                }

                fout.close();
            }


            cout << "Trial #" << mcCount << " completed." << endl;
            mcCount++;

            // Clear Variables for next trial
            storeRow.resize(0,0);

            u1.clear();
            u2.clear();
            u3.clear();
            u4.clear();
            u5.clear();
            u6.clear();
            controls.clear();

            x.clear();
            y.clear();
            z.clear();
            dx.clear();
            dy.clear();
            dz.clear();
            sq.clear();
            vq.clear();
            dw.clear();
            states.clear();

            u_min.clear();
            u_max.clear();
            u_init.clear();
            x0_min.clear();
            x0_max.clear();
            xf_min.clear();
            xf_max.clear();
            x_init.clear();

            V.clear();
            X.clear();
            U.clear();
            v_min.clear();
            v_max.clear();
            v_init.clear();

            J.clear();
            Q.clear();
            R.clear();
            xd.clear();

            gAlgebraic.clear();
            k1.clear();
            k2.clear();
            k3.clear();
            k4.clear();
            stNextRK4.clear();
            stNextEuler.clear();
            Rtc_k1.clear();
            Rtc_k2.clear();
            Rtc_k3.clear();
            Rtc_k4.clear();

            nlp.clear();
            opts.clear();
            solvers.clear();
            arg.clear();
            res.clear();

            Storex0.resize(0,0);
            x0.resize(0,0);
            xs.resize(0,0);
            xx.resize(0,0);
            xx1.resize(0,0);
            X0.resize(0,0);
            u0.resize(0,0);
            uwu.resize(0,0);
            u_cl.resize(0,0);

            MPCstates.clear();
            MPCcontrols.clear();
            sol.clear();

        }

        cout << "# Converged: " << numConverged << endl;
    }

    return 0;
}



//////////////////////////////////////////////////////////////////////////////
// Function Name: Skew
// Description: This function is used to create a skew matrix given a vector
// Inputs: VectorXd f - vector
// Outputs: MatrixXd S - skew matrix
//////////////////////////////////////////////////////////////////////////////
MatrixXd Skew(VectorXd f)
{
    MatrixXd S(3,3);
    S << 0, -f(2), f(1),
            f(2), 0, -f(0),
            -f(1), f(0), 0;

    return S;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: q2R
// Description: This function is used to convert from quaternion to rotation matrix
// Inputs: MatrixXd q - quaternion
// Outputs: MatrixXd R - roatation matrix
//////////////////////////////////////////////////////////////////////////////
MatrixXd q2R(MatrixXd q){
    MatrixXd R(3,3);
    double sq = q(0);
    VectorXd vq(3);
    vq << q(1),q(2),q(3);

    R(0,0) = pow(sq,2) + pow(vq(0),2) - pow(vq(1),2) - pow(vq(2),2);
    R(0,1) = 2*(vq(0)*vq(1)-sq*vq(2));
    R(0,2) = 2*(vq(0)*vq(2)+sq*vq(1));

    R(1,0) = 2*(vq(0)*vq(1)+sq*vq(2));
    R(1,1) = pow(sq,2) - pow(vq(0),2) + pow(vq(1),2) - pow(vq(2),2);
    R(1,2) = 2*(vq(1)*vq(2)-sq*vq(0));

    R(2,0) =  2*(vq(0)*vq(2)-sq*vq(2));
    R(2,1) =  2*(vq(1)*vq(2)+sq*vq(0));
    R(2,2) =  pow(sq,2) - pow(vq(0),2) - pow(vq(1),2) + pow(vq(2),2);

    return R;
};

//////////////////////////////////////////////////////////////////////////////
// Function Name: q2R
// Description: This function is used to convert from quaternion to rotation matrix
// Inputs: MX st - current state
// Outputs: MX RR - roatation matrix
//////////////////////////////////////////////////////////////////////////////
SX q2R(SX st)
{
    SX sq = st(6);
    SX v1 = st(7);
    SX v2 = st(8);
    SX v3 = st(9);
    SX vq = vertcat(v1,v2,v3);
//    MX RR = MX::sym("RR",3,3);

    SX r00 = pow(sq,2) + pow(vq(0),2) - pow(vq(1),2) - pow(vq(2),2);
    SX r01 = 2*(vq(0)*vq(1)-sq*vq(2));
    SX r02 = 2*(vq(0)*vq(2)+sq*vq(1));
    SX row0 = horzcat(r00,r01,r02);

    SX r10= 2*(vq(0)*vq(1)+sq*vq(2));
    SX r11 = pow(sq,2) - pow(vq(0),2) + pow(vq(1),2) - pow(vq(2),2);
    SX r12= 2*(vq(1)*vq(2)-sq*vq(0));
    SX row1 = horzcat(r10,r11,r12);

    SX r20 =  2*(vq(0)*vq(2)-sq*vq(2));
    SX r21 =  2*(vq(1)*vq(2)+sq*vq(0));
    SX r22 =  pow(sq,2) - pow(vq(0),2) - pow(vq(1),2) + pow(vq(2),2);
    SX row2 = horzcat(r20,r21,r22);

    SX RR = vertcat(row0, row1, row2);

    return RR;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: f
// Description: This function is used to calculate the time derivative of our system
// Inputs: MX st - current state, MX con - current control action, MX Rtc - Rotation matrix,
//         double n - mean motion of target satellite, double mc - mass of chaser satellite
// Outputs: MatrixXd xDot - time derivative of the current state
//////////////////////////////////////////////////////////////////////////////
SX f(SX st, SX con, SX Rtc , double n, double mc)
{
    SX x = st(0);
    SX y = st(1);
    SX z = st(2);
    SX dx = st(3);
    SX dy = st(4);
    SX dz = st(5);
    SX sq = st(6);
    SX v1 = st(7);
    SX v2 = st(8);
    SX v3 = st(9);
    SX dw1 = st(10);
    SX dw2 = st(11);
    SX dw3 = st(12);

    SX vq = vertcat(v1,v2,v3);
    SX dw = vertcat(dw1,dw2,dw3);

    SX u1 = con(0);
    SX u2 = con(1);
    SX u3 = con(2);
    SX u4 = con(3);
    SX u5 = con(4);
    SX u6 = con(5);

    SX thrust = vertcat(u1,u2,u3);
    SX torque = vertcat(u4,u5,u6);

    //MX Rtc = q2R(sq,vq);
    SX f = mtimes(Rtc,thrust);
    SX Jd = SX::zeros(3,3); // Moment of inertia for the chaser
    Jd(0,0) = 0.2734;
    Jd(1,1) = 0.2734;
    Jd(2,2) = 0.3125;

    SX wt = SX::zeros(3,1); // angular velocity of the target
    wt(2) = n;

    SX Kd = solve(Jd,SX::eye(Jd.size1())); // Kd = Jd^-1
    SX Ko = mtimes(mtimes(Rtc,Kd),Rtc);

    SX rhs = vertcat(dx, dy, dz,
                     3* pow(n,2)*x+2*n*dy+f(0)/mc,
                     -2*n*dx+f(1)/mc,
                     -pow(n,2)*z+f(2)/mc );

    rhs = vertcat(rhs,
                  0.5*dot(vq,dw),
                  -0.5 * mtimes( ((sq * SX::eye(3)) + skew(vq)), dw),
                  mtimes(skew(dw),wt) + mtimes(Ko,torque)
                  - mtimes(Ko, ( mtimes(skew(dw), mtimes( mtimes(Jd,Rtc.T()), dw))
                                 + mtimes(skew(dw), mtimes( mtimes(Jd,Rtc.T()), wt))
                                 + mtimes(skew(wt), mtimes( mtimes(Jd,Rtc.T()), dw))
                                 + mtimes(skew(wt), mtimes( mtimes(Jd,Rtc.T()), wt)))));
    return rhs;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: f
// Description: This function is used to calculate the time derivative of our system
// Inputs: MatrixXd st - current state, MatrixXd con - current control action
// Outputs: MatrixXd xDot - time derivative of the current state
//////////////////////////////////////////////////////////////////////////////
MatrixXd f(MatrixXd st, MatrixXd con, double n, double mc)
{
    double x = st(0);
    double y = st(1);
    double z = st(2);
    double dx = st(3);
    double dy = st(4);
    double dz = st(5);
    double sq = st(6);
    double v1 = st(7);
    double v2 = st(8);
    double v3 = st(9);
    double dw1 = st(10);
    double dw2 = st(11);
    double dw3 = st(12);

    VectorXd dw(3);
    dw << dw1,dw2,dw3;
    VectorXd vq(3);
    vq << v1,v2,v3;
    VectorXd q(4);
    q << sq,vq;

    double u1 = con(0);
    double u2 = con(1);
    double u3 = con(2);
    double u4 = con(3);
    double u5 = con(4);
    double u6 = con(5);

    VectorXd thrust(3);
    thrust << u1,u2,u3;
    VectorXd torque(3);
    torque << u4,u5,u6;

    MatrixXd Rtc = q2R(q);
    VectorXd f = Rtc*thrust;
    Matrix3d Jd = Matrix3d::Zero(); // Moment of inertia for the chaser
    Jd(0,0) = 0.2734;
    Jd(1,1) = 0.2734;
    Jd(2,2) = 0.3125;

    VectorXd wt(3); // angular velocity of the target
    wt(0) = 0;
    wt(1) = 0;
    wt(2) = n;

    MatrixXd Kd = Jd.inverse(); // Kd = Jd^-1
    MatrixXd Ko = Rtc*Kd*Rtc;

    MatrixXd xDot(13,1);
    xDot << dx, dy, dz,
            3* pow(n,2)*x + 2*n*dy + f(0)/mc,
            -2*n*dx + f(1)/mc,
            -pow(n,2)*z + f(2)/mc,
            0.5*vq.dot(dw),
            -0.5*((sq * MatrixXd::Identity(3,3)) + Skew(vq)) * dw,
            Skew(dw)*wt + Ko*torque
            - Ko*( Skew(dw)*(Jd*Rtc.transpose()*dw)
                   + Skew(dw)*(Jd*Rtc.transpose()*wt)
                   + Skew(wt)*(Jd*Rtc.transpose()*dw)
                   + Skew(wt)*(Jd*Rtc.transpose()*wt) );
    return xDot;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: shift
// Description: This function is used to shift our MPC states and control inputs
//              in time so that we can re-initialize our optimization problem
//              with the new current state of the system
// Inputs: N - Prediction Horizon, ts - sampling time, x0 - initial state,
//             uwu - optimal control sequence from NLP, u0 - shifted
//             control sequence, n - mean motion of target satellite,
//             mc - mass of chaser satellite
// Outputs: None
//////////////////////////////////////////////////////////////////////////////
void shiftRK4(int N, double ts, MatrixXd& x0, MatrixXd uwu, MatrixXd& u0, double n, double mc)
{
    // Shift State
    MatrixXd st = x0;
    MatrixXd con = uwu.col(0);

    MatrixXd k1 = f(st,con,n,mc);
    MatrixXd k2 = f(st + (ts/2)*k1,con,n,mc);
    MatrixXd k3 = f(st + (ts/2)*k2,con,n,mc);
    MatrixXd k4 = f(st + ts*k3,con,n,mc);

    st = st + ts/6*(k1 + 2*k2 + 2*k3 + k4);
    x0 = st;

    // Shift Control
    std::vector<int> ind(N-1) ; // vector with N-1 integers to be filled
    std::iota (std::begin(ind), std::end(ind), 1); // fill vector with N-1 integers starting at 1
    u0 = uwu(Eigen::placeholders::all,ind); // assign u0 with columns 1-(N-1) of uwu
    u0.conservativeResize(u0.rows(), u0.cols()+1);
    u0.col(u0.cols()-1) = uwu(Eigen::placeholders::all,Eigen::placeholders::last); // copy last column and append it
}

