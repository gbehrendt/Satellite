#include <iostream>
#include <omp.h>
#include <cstdio>
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
MX q2R(MX);
MatrixXd f(MatrixXd, MatrixXd, double, double);
MX f(MX, MX, MX, double, double);
void shiftRK4(int, double, MatrixXd &, MatrixXd,MatrixXd &, double, double);

int main()
{
    // File pointer
    fstream fin;

    // Open an existing file
    fin.open("/home/gbehrendt/CLionProjects/finalSatellite/initialConditions.csv", ios::in);
    if (fin.is_open()) {
        cout << "File opened successfully :)" << endl;
    } else {
        cerr << "File not opened :(" << endl;
        return -1;
    }

    // Read the Data from the file as String Vector
    std::vector<double> row;
    string item;
    int numConverged = 0;
    int numConditions = 200;

    double initConditions[numConditions][14];

    // Read in initial conditions
    int ii = 0;
    while (getline(fin, item)) {
        row.clear();
        istringstream line(item);
        while (getline(line, item, ',')) {
            row.push_back(stod(item));
        }
        for(int jj=0;jj<14;jj++)
        {
            initConditions[ii][jj]=row[jj];
        }
        ii++;
    }



    // Declare states + controls
    SX u1 = SX::sym("u1"); // x thrust
    SX u2 = SX::sym("u2"); // y thrust
    SX u3 = SX::sym("u3"); // z thrust
    SX u4 = SX::sym("u4"); // x torque
    SX u5 = SX::sym("u5"); // y torque
    SX u6 = SX::sym("u6"); // z torque

    SX controls = vertcat(u1,u2,u3,u4,u5,u6);

    MX x = MX::sym("x");
    MX y = MX::sym("y");
    MX z = MX::sym("z");
    MX dx = MX::sym("dx");
    MX dy = MX::sym("dy");
    MX dz = MX::sym("dz");
    MX sq = MX::sym("sq");
    MX vq = MX::sym("vq",3,1);
    MX dw = MX::sym("dw",3,1);

    MX states = vertcat(x,y,z,dx,dy,dz);
    states = vertcat(states,sq,vq,dw);

    // Number of differential states
    const int numStates = states.size1();

    // Number of controls
    const int numControls = controls.size1();

    // Bounds and initial guess for the control
    double thrustMax = 1e-2;
    double thrustMin = -thrustMax;
    double torqueMax = 1e-4;
    double torqueMin = -torqueMax;
    std::vector<double> u_min =  { thrustMin, thrustMin, thrustMin, torqueMin, torqueMin, torqueMin };
    std::vector<double> u_max  = { thrustMax, thrustMax, thrustMax, torqueMax, torqueMax, torqueMax };
    std::vector<double> u_init = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Initial Satellite Conditions + Problem Parameters
    double tPeriod = 92.68 * 60; // ISS orbital period (seconds)
    double n = -2*M_PI/tPeriod; // Mean motion of ISS (rad/s)
    double mc = 12; // mass of the chaser

    std::vector<double> x_min  = { -inf, -inf, -inf, -inf, -inf, -inf, 0, 0, 0, 0, -inf, -inf, -inf };
    std::vector<double> x_max  = { inf, inf, inf, inf, inf, inf, 1, 1, 1, 1, inf, inf, inf };

    std::vector<double> xf_min = { -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf };
    std::vector<double> xf_max = { inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf };
    std::vector<double> x_init = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Tunable Parameters
    bool writeToFile = false; // choose to write to file or not
    string hessianApprox = "limited-memory"; // Choices: "limited-memory" or "exact" ("limited-memory" runs slightly faster, but "exact" works better for convergence i.e. less MPC loops)
    string constraintType = "Euler"; // Choices: "RK4" or "Euler"
    const int N = 100; // Prediction Horizon
    double ts = 10.0; // sampling period
    int maxIterArr[] = {5,10,20,50,100};

    double posCost = 1e10;
    double velCost = 1e2;
    double quatCost = 1e12;
    double angularCost = 1e12;
    double thrustCost = 1e10;
    double torqueCost = 1e-10;

    // Total number of NLP variables
    const int numVars = numStates*(N+1) + numControls*N;

    // Declare variable vector for the NLP
    MX V = MX::sym("V",numVars);

    // Initialize Objective Function and Weighting Matrices
    MX J = 0; // Objective Function
    MX Q = MX::zeros(numStates,numStates);
    MX R = MX::zeros(numControls,numControls);

    Q(0,0) = posCost; // xPos
    Q(1,1) = posCost; // yPos
    Q(2,2) = posCost; // zPos
    Q(3,3) = velCost; // dx
    Q(4,4) = velCost; // dy
    Q(5,5) = velCost; // dz
    Q(6,6) = quatCost; // sq
    Q(7,7) = quatCost; // vq
    Q(8,8) = quatCost; // vq
    Q(9,9) = quatCost; // vq
    Q(10,10) = angularCost; // dw
    Q(11,11) = angularCost; // dw
    Q(12,12) = angularCost; // dw

    R(0,0) = thrustCost;
    R(1,1) = thrustCost;
    R(2,2) = thrustCost;
    R(3,3) = torqueCost;
    R(4,4) = torqueCost;
    R(5,5) = torqueCost;

    MX xd = MX::zeros(numStates);
    xd(0) = 0.0; xd(1) = 0.0; xd(2) = 0.0;
    xd(3) = 0.0; xd(4) = 0.0; xd(5) = 0.0;
    xd(6) = 1.0; xd(7) = 0.0; xd(8) = 0.0; xd(9) = 0.0;
    xd(10) = 0.0; xd(11) = 0.0; xd(12) = 0.0;

    // Offset in V
    int offset=0;

    // State at each shooting node and control for each shooting interval
    std::vector<MX> X, U;
    for(int k=0; k<N; ++k){
        // Local state
        X.push_back( V.nz(Slice(offset,offset + numStates)));
        offset += numStates;

        // Local control
        U.push_back( V.nz(Slice(offset,offset + numControls)));
        offset += numControls;
    }
    // State at end
    X.push_back(V.nz(Slice(offset,offset+numStates)));
    offset += numStates;

    // Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
    casadi_assert(offset==numVars, "");

    //Constraint function and bounds
    std::vector<MX> gAlgebraic;
    MX k1,k2,k3,k4,stNextRK4, stNextEuler;
    MX Rtc_k1, Rtc_k2, Rtc_k3, Rtc_k4;

    if (constraintType == "RK4")
    {
        // Loop over shooting nodes
        for(int k=0; k<N; ++k){
            // algebraic definition of constraints
            Rtc_k1 = q2R(X[k]);
            k1 = f(X[k],U[k],Rtc_k1,n,mc);

            Rtc_k2 = q2R(X[k] + (ts/2)*k1);
            k2 = f(X[k] + (ts/2)*k1,U[k],Rtc_k2,n,mc);

            Rtc_k3 = q2R(X[k] + (ts/2)*k2);
            k3 = f(X[k] + (ts/2)*k2,U[k],Rtc_k3,n,mc);

            Rtc_k4 = q2R(X[k] + ts*k3);
            k4 = f(X[k] + ts*k3,U[k],Rtc_k4,n,mc);

            stNextRK4 = X[k] + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);

            // Save continuity constraints
            gAlgebraic.push_back( stNextRK4 - X[k+1] );

            // Add objective function contribution
            J += mtimes(mtimes((X[k]-xd).T(),Q),(X[k]-xd)) + mtimes(mtimes(U[k].T(),R),U[k]);
        }
    }
    else if(constraintType == "Euler")
    {
        for(int k=0; k<N; ++k){
            // algebraic definition of constraints
            Rtc_k1 = q2R(X[k]);
            stNextEuler = X[k] + ts*f(X[k],U[k],Rtc_k1,n,mc);

            // Save continuity constraints
            gAlgebraic.push_back( stNextEuler - X[k+1] );

            // Add objective function contribution
            J += mtimes(mtimes((X[k]-xd).T(),Q),(X[k]-xd)) + mtimes(mtimes(U[k].T(),R),U[k]);
        }
    }

    // NLP
    MXDict nlp = {{"x", V}, {"f", J}, {"g", vertcat(gAlgebraic)}};


    int numUsedThreads = 10;


    // Create an NLP solver and buffers
    std::map<std::string, DM> Arg[numUsedThreads];
    std::map<std::string, DM> Sol[numUsedThreads];
    std::vector<double> v_min[numUsedThreads],v_max[numUsedThreads],v_init[numUsedThreads];

    int numBatches = 10;
    int trialsPerBatch = numConditions/numBatches;

    Dict Opts[trialsPerBatch];
    Function solverArr[trialsPerBatch];
    Function solver;



    int maxIterLength = sizeof(maxIterArr)/sizeof(maxIterArr[0]);

    for(int kk=0; kk < maxIterLength; kk++)
    {
        int batchCount = 0;
        while(batchCount < numBatches)
        {
            string timePath = "/home/gbehrendt/CLionProjects/untitled/parallel/Timing3/" + constraintType + "/" + hessianApprox + "/ts" + to_string(to_int(ts)) + "/maxIter" + to_string(maxIterArr[kk]) + "/trial";

            // Set options
            printf("Filling Opts and Solver arrays for Batch #%d \n",batchCount);
            for(int i=0;i<trialsPerBatch;i++)
            {
                cout << i << "  " << timePath + to_string(batchCount*trialsPerBatch + i) + ".csv" << endl;
                if(writeToFile == true)
                {
                    Opts[i]["ipopt.tol"] = 1e-5;
                    Opts[i]["ipopt.max_iter"] = maxIterArr[kk];
                    Opts[i]["ipopt.hessian_approximation"] = hessianApprox; // for no max iterations change from "limited-memory" to "exact"
                    Opts[i]["ipopt.print_level"] = 0;
                    Opts[i]["ipopt.acceptable_tol"] = 1e-8;
                    Opts[i]["ipopt.acceptable_obj_change_tol"] = 1e-6;
                    Opts[i]["expand"] = false;
                    Opts[i]["ipopt.file_print_level"] = 3;
                    Opts[i]["ipopt.output_file"] = timePath + to_string(batchCount*trialsPerBatch + i) + ".csv";
                    Opts[i]["ipopt.print_timing_statistics"] = "yes";
                }
                else if(writeToFile == false)
                {
                    Opts[i]["ipopt.tol"] = 1e-5;
                    Opts[i]["ipopt.max_iter"] = maxIterArr[kk];
                    Opts[i]["ipopt.hessian_approximation"] = hessianApprox; // for no max iterations change from "limited-memory" to "exact"
                    Opts[i]["ipopt.print_level"] = 0;
                    Opts[i]["ipopt.acceptable_tol"] = 1e-8;
                    Opts[i]["ipopt.acceptable_obj_change_tol"] = 1e-6;
                    Opts[i]["expand"] = false;
                }
                solverArr[i] = nlpsol("nlpsol", "ipopt", nlp, Opts[i]);
            }
            cout << "Done filling Opts and Solver arrays." << endl;

            int startInd = batchCount*trialsPerBatch;

            while(startInd <= batchCount*trialsPerBatch+trialsPerBatch-numUsedThreads)
            {
                #pragma omp parallel for num_threads(numUsedThreads)
                for (int i = 0; i < numUsedThreads; i++)
                {
                    int tid = omp_get_thread_num();
                    int trialNum = initConditions[startInd + i][0];
                    string path = "/home/gbehrendt/CLionProjects/untitled/parallel/Results3/" + constraintType + "/" +
                                  hessianApprox + "/ts" + to_string(to_int(ts)) + "/maxIter" +
                                  to_string(maxIterArr[kk]) + "/trial" + to_string(trialNum) + ".csv";

                    Eigen::MatrixXd x0(13, 1);
                    for (int j = 0; j < 13; j++)
                    {
                        x0(j) = initConditions[startInd + i][j + 1];
                    }
                    //            printf("The thread %d  executes i = %d, trial # : %d \n", tid, i, trialNum);
                    printf("The thread %d  executes i = %d, trial #%d, x0 = %f %f %f %f %f %f %f %f %f %f %f %f %f \n",
                           tid, i, trialNum, x0(0), x0(1), x0(2), x0(3), x0(4), x0(5), x0(6), x0(7), x0(8), x0(9),
                           x0(10), x0(11), x0(12));

                    double xPos = x0(0);
                    double yPos = x0(1);
                    double zPos = x0(2);
                    double xVel = x0(3);
                    double yVel = x0(4);
                    double zVel = x0(5);
                    std::vector<double> q0 = { x0(6), x0(7), x0(8), x0(9) }; // Normalized [0.5,0.3,0.2,0.2]
                    std::vector<double> dw0 = { x0(10), x0(11), x0(12)};

                    // Bounds and initial guess for the state
                    std::vector<double> x0_min = {  xPos, yPos, zPos, xVel, yVel, zVel }; // initial position and velocity
                    x0_min.insert(x0_min.end(),q0.begin(),q0.end()); // append initial quaternion
                    x0_min.insert(x0_min.end(),dw0.begin(),dw0.end()); // append initial angular velocity

                    std::vector<double> x0_max = {  xPos, yPos, zPos, xVel, yVel, zVel };
                    x0_max.insert(x0_max.end(),q0.begin(),q0.end()); // append initial quaternion
                    x0_max.insert(x0_max.end(),dw0.begin(),dw0.end()); // append initial angular velocity

                    // NLP variable bounds and initial guess
                    // State at each shooting node and control for each shooting interval
                    for(int k=0; k<N; ++k){
                        // Local state
                        if(k==0){
                            v_min[i].insert(v_min[i].end(), x0_min.begin(), x0_min.end());
                            v_max[i].insert(v_max[i].end(), x0_max.begin(), x0_max.end());
                        } else {
                            v_min[i].insert(v_min[i].end(), x_min.begin(), x_min.end());
                            v_max[i].insert(v_max[i].end(), x_max.begin(), x_max.end());
                        }
                        v_init[i].insert(v_init[i].end(), x_init.begin(), x_init.end());

                        // Local control
                        v_min[i].insert(v_min[i].end(), u_min.begin(), u_min.end());
                        v_max[i].insert(v_max[i].end(), u_max.begin(), u_max.end());
                        v_init[i].insert(v_init[i].end(), u_init.begin(), u_init.end());
                    }

                    // State at end
                    v_min[i].insert(v_min[i].end(), xf_min.begin(), xf_min.end());
                    v_max[i].insert(v_max[i].end(), xf_max.begin(), xf_max.end());
                    v_init[i].insert(v_init[i].end(), x_init.begin(), x_init.end());

                    // Bounds and initial guess
                    Arg[i]["lbx"] = v_min[i];
                    Arg[i]["ubx"] = v_max[i];
                    Arg[i]["lbg"] = 0;
                    Arg[i]["ubg"] = 0;
                    Arg[i]["x0"] = v_init[i];

                    //---------------------//
                    //      MPC Loop       //
                    //---------------------//
                    Eigen::MatrixXd Storex0(numStates,1);
                    Storex0 = x0;

                    // Define docking configuration
                    Eigen::MatrixXd xs(numStates,1);
                    xs << 0,0,0,0,0,0,1,0,0,0,0,0,0;

                    Eigen::MatrixXd xx(numStates, N+1);
                    xx.col(0) = x0;
                    Eigen::MatrixXd xx1(numStates, N+1);
                    Eigen::MatrixXd X0(numStates,N+1);
                    Eigen::MatrixXd u0;
                    Eigen::MatrixXd uwu(numControls,N);
                    Eigen::MatrixXd u_cl(numControls,N);

                    vector<vector<double> > MPCstates(numStates);
                    vector<vector<double> > MPCcontrols(numControls);


                    for(int j=0; j<numStates; j++)
                    {
                        MPCstates[j].push_back(x0(j));
                    }

                    // Start MPC
                    int iter = 0;
                    double epsilon = 1e-3;
                    double infNorm = 10;
                    while( infNorm > epsilon  && iter < N && infNorm < 100)
                    {
                        // Solve NLP
                        Sol[i] = solverArr[startInd - batchCount*trialsPerBatch + i](Arg[i]);

                        std::vector<double> V_opt(Sol[i].at("x"));

                        Eigen::MatrixXd V = Eigen::Map<Eigen::Matrix<double, 1913, 1> >(V_opt.data()); // N=100
                        // Store Solution
                        for(int ii=0; ii<=N; ++ii)
                        {
                            xx1(0,ii) = V(ii*(numStates+numControls));
                            xx1(1,ii) = V(1+ii*(numStates+numControls));
                            xx1(2,ii) = V(2+ii*(numStates+numControls));
                            xx1(3,ii) = V(3+ii*(numStates+numControls));
                            xx1(4,ii) = V(4+ii*(numStates+numControls));
                            xx1(5,ii) = V(5+ii*(numStates+numControls));
                            xx1(6,ii) = V(6+ii*(numStates+numControls));
                            xx1(7,ii) = V(7+ii*(numStates+numControls));
                            xx1(8,ii) = V(8+ii*(numStates+numControls));
                            xx1(9,ii) = V(9+ii*(numStates+numControls));
                            xx1(10,ii) = V(10+ii*(numStates+numControls));
                            xx1(11,ii) = V(11+ii*(numStates+numControls));
                            xx1(12,ii) = V(12+ii*(numStates+numControls));
                            if(ii < N)
                            {
                                uwu(0,ii)= V(numStates + ii*(numStates+numControls));
                                uwu(1,ii) = V(1+numStates + ii*(numStates+numControls));
                                uwu(2,ii) = V(2+numStates + ii*(numStates+numControls));
                                uwu(3,ii) = V(3+numStates + ii*(numStates+numControls));
                                uwu(4,ii) = V(4+numStates + ii*(numStates+numControls));
                                uwu(5,ii) = V(5+numStates + ii*(numStates+numControls));
                            }
                        }

                        // Get solution Trajectory
                        u_cl.col(iter) = uwu.col(0); // Store first control action from optimal sequence

                        // Apply control and shift solution
                        shiftRK4(N,ts,x0,uwu,u0,n,mc);
                        xx(Eigen::placeholders::all,iter+1)=x0;

                        // Shift trajectory to initialize next step
                        std::vector<int> ind(N) ; // vector with N-1 integers to be filled
                        std::iota (std::begin(ind), std::end(ind), 1); // fill vector with N integers starting at 1
                        X0 = xx1(Eigen::placeholders::all,ind); // assign X0 with columns 1-(N) of xx1
                        X0.conservativeResize(X0.rows(), X0.cols()+1);
                        X0.col(X0.cols()-1) = xx1(Eigen::placeholders::all,Eigen::placeholders::last);


                        for(int j=0; j<numStates; j++)
                        {
                            MPCstates[j].push_back(x0(j));
                        }

                        for(int j=0; j<numControls; j++)
                        {
                            MPCcontrols[j].push_back(u_cl(j,iter));
                        }

                        // Re-initialize Problem Parameters
                        v_min[i].erase(v_min[i].begin(),v_min[i].begin()+numStates);
                        v_min[i].insert(v_min[i].begin(),x0(12));
                        v_min[i].insert(v_min[i].begin(),x0(11));
                        v_min[i].insert(v_min[i].begin(),x0(10));
                        v_min[i].insert(v_min[i].begin(),x0(9));
                        v_min[i].insert(v_min[i].begin(),x0(8));
                        v_min[i].insert(v_min[i].begin(),x0(7));
                        v_min[i].insert(v_min[i].begin(),x0(6));
                        v_min[i].insert(v_min[i].begin(),x0(5));
                        v_min[i].insert(v_min[i].begin(),x0(4));
                        v_min[i].insert(v_min[i].begin(),x0(3));
                        v_min[i].insert(v_min[i].begin(),x0(2));
                        v_min[i].insert(v_min[i].begin(),x0(1));
                        v_min[i].insert(v_min[i].begin(),x0(0));

                        v_max[i].erase(v_max[i].begin(),v_max[i].begin()+numStates);
                        v_max[i].insert(v_max[i].begin(),x0(12));
                        v_max[i].insert(v_max[i].begin(),x0(11));
                        v_max[i].insert(v_max[i].begin(),x0(10));
                        v_max[i].insert(v_max[i].begin(),x0(9));
                        v_max[i].insert(v_max[i].begin(),x0(8));
                        v_max[i].insert(v_max[i].begin(),x0(7));
                        v_max[i].insert(v_max[i].begin(),x0(6));
                        v_max[i].insert(v_max[i].begin(),x0(5));
                        v_max[i].insert(v_max[i].begin(),x0(4));
                        v_max[i].insert(v_max[i].begin(),x0(3));
                        v_max[i].insert(v_max[i].begin(),x0(2));
                        v_max[i].insert(v_max[i].begin(),x0(1));
                        v_max[i].insert(v_max[i].begin(),x0(0));

                        // Re-initialize initial guess
                        v_init[i] = V_opt;
                        v_init[i].erase(v_init[i].begin(),v_init[i].begin()+(numStates + numControls));
                        std::vector<double> finalStates;
                        copy(v_init[i].end()-(numStates+numControls),v_init[i].end(),back_inserter(finalStates));
                        v_init[i].insert(v_init[i].end(),finalStates.begin(),finalStates.end());

                        Arg[i]["lbx"] = v_min[i];
                        Arg[i]["ubx"] = v_max[i];
                        Arg[i]["x0"] = v_init[i];

                        infNorm = max((x0-xs).lpNorm<Eigen::Infinity>(),u_cl.col(iter).lpNorm<Eigen::Infinity>()); // l-infinity norm of current state and control
                        //printf("-------------------- Trial #%d infNorm: %f --------------------\n",trialNum,infNorm);
                        iter++;
                    }

                    string converged;

                    if(infNorm < epsilon)
                    {
                        converged = "yes";
                        numConverged += 1;
                        printf("*************************** maxIter: %d Trial #%d = SUCCESS!!! *************************** \n",maxIterArr[kk],trialNum);
                    }
                    else
                    {
                        converged = "no";
                    }

                    if(writeToFile == true)
                    {
                        const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
                        ofstream fout; // declare fout variable

                        fout.open(path, std::ofstream::out | std::ofstream::trunc ); // open file to write to

                        fout << "x (km),y (km),z (km),xdot (km/s),ydot (km/s),zdot (km/s),sq,v1,v2,v3,dw1 (rad/s),dw2 (rad/s),dw3 (rad/s),thrust1 (N),thrust2 (N),thrust3 (N),tau1 (rad/s^2),tau2 (rad/s^2),tau3 (rad/s^2),x0,Maximum Iterations,ts,N,MPC Loops,posCost,velCost,quatCost,angualarCost,thrustCost,torqueCost,"
                                "thrustMax,torqueMax,Constraint Type,Trial,converged?,infNorm" <<endl;

                        for(int j=0; j < iter; j++)
                        {
                            if(j==0)
                            {
                                for(int i=0; i < (MPCstates.size() + MPCcontrols.size()); i++)
                                {
                                    if (i < numStates) {
                                        fout << MPCstates[i][j] << ",";
                                    }
                                    else if (i < numStates + numControls) {
                                        fout << MPCcontrols[i - numStates][j] << ",";
                                    }
                                }
                                fout<< Storex0(j)<<","<<maxIterArr[kk] <<","<<ts<<","<<N<<","<<iter<<","<<posCost<<","<<velCost<<","<<quatCost<<","<<angularCost<<","<<thrustCost <<","<<torqueCost<<","
                                    <<thrustMax<<","<<torqueMax<<","<<constraintType<<","<<trialNum<<","<<converged<<","<<infNorm;
                            }
                            else{
                                for(int i=0; i < (MPCstates.size() + MPCcontrols.size()); i++)
                                {
                                    if (i < numStates) {
                                        fout << MPCstates[i][j] << ",";
                                    }
                                    else if (i < numStates + numControls) {
                                        fout << MPCcontrols[i - numStates][j] << ",";
                                    }
                                }
                                if(j<numStates){
                                    fout<<Storex0(j)<<",";
                                }
                            }
                            fout<<endl;
                        }

                        fout.close();
                    }


                    q0.clear();
                    dw0.clear();
                    x0_min.clear();
                    x0_max.clear();
                    v_min[i].clear();
                    v_max[i].clear();
                    v_init[i].clear();


                }

                cout << "---------------- NEXT ROUND OF CONDITIONS ----------------" << endl;
                startInd += numUsedThreads;
            }
            batchCount += 1;
        }

    }
    return 0;
}

MatrixXd Skew(VectorXd f)
{
    MatrixXd S(3,3);
    S << 0, -f(2), f(1),
            f(2), 0, -f(0),
            -f(1), f(0), 0;

    return S;
}

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

MX q2R(MX st)
{
    MX sq = st(6);
    MX v1 = st(7);
    MX v2 = st(8);
    MX v3 = st(9);
    MX vq = vertcat(v1,v2,v3);
//    MX RR = MX::sym("RR",3,3);

    MX r00 = pow(sq,2) + pow(vq(0),2) - pow(vq(1),2) - pow(vq(2),2);
    MX r01 = 2*(vq(0)*vq(1)-sq*vq(2));
    MX r02 = 2*(vq(0)*vq(2)+sq*vq(1));
    MX row0 = horzcat(r00,r01,r02);

    MX r10= 2*(vq(0)*vq(1)+sq*vq(2));
    MX r11 = pow(sq,2) - pow(vq(0),2) + pow(vq(1),2) - pow(vq(2),2);
    MX r12= 2*(vq(1)*vq(2)-sq*vq(0));
    MX row1 = horzcat(r10,r11,r12);

    MX r20 =  2*(vq(0)*vq(2)-sq*vq(2));
    MX r21 =  2*(vq(1)*vq(2)+sq*vq(0));
    MX r22 =  pow(sq,2) - pow(vq(0),2) - pow(vq(1),2) + pow(vq(2),2);
    MX row2 = horzcat(r20,r21,r22);

    MX RR = vertcat(row0, row1, row2);

    return RR;
}

MX f(MX st, MX con, MX Rtc , double n, double mc)
{
    MX x = st(0);
    MX y = st(1);
    MX z = st(2);
    MX dx = st(3);
    MX dy = st(4);
    MX dz = st(5);
    MX sq = st(6);
    MX v1 = st(7);
    MX v2 = st(8);
    MX v3 = st(9);
    MX dw1 = st(10);
    MX dw2 = st(11);
    MX dw3 = st(12);

    MX vq = vertcat(v1,v2,v3);
    MX dw = vertcat(dw1,dw2,dw3);

    MX u1 = con(0);
    MX u2 = con(1);
    MX u3 = con(2);
    MX u4 = con(3);
    MX u5 = con(4);
    MX u6 = con(5);

    MX thrust = vertcat(u1,u2,u3);
    MX torque = vertcat(u4,u5,u6);

    //MX Rtc = q2R(sq,vq);
    MX f = mtimes(Rtc,thrust);
    MX Jd = MX::zeros(3,3); // Moment of inertia for the chaser
    Jd(0,0) = 0.2734;
    Jd(1,1) = 0.2734;
    Jd(2,2) = 0.3125;

    MX wt = MX::zeros(3,1); // angular velocity of the target
    wt(2) = n;

    MX Kd = solve(Jd,MX::eye(Jd.size1())); // Kd = Jd^-1
    MX Ko = mtimes(mtimes(Rtc,Kd),Rtc);

    MX rhs = vertcat(dx, dy, dz,
                     3* pow(n,2)*x+2*n*dy+f(0)/mc,
                     -2*n*dx+f(1)/mc,
                     -pow(n,2)*z+f(2)/mc );
    rhs = vertcat(rhs,
                  0.5*dot(vq,dw),
                  -0.5 * mtimes( ((sq * MX::eye(3)) + skew(vq)), dw),
                  mtimes(skew(dw),wt) + mtimes(Ko,torque)
                  - mtimes(Ko, ( mtimes(skew(dw), mtimes( mtimes(Jd,Rtc.T()), dw))
                                 + mtimes(skew(dw), mtimes( mtimes(Jd,Rtc.T()), wt))
                                 + mtimes(skew(wt), mtimes( mtimes(Jd,Rtc.T()), dw))
                                 + mtimes(skew(wt), mtimes( mtimes(Jd,Rtc.T()), wt))))
    );

    return rhs;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: f
// Description: This function is used to implement the dynamics of our system
//              once a control action is implemented
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
