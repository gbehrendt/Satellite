#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <casadi/casadi.hpp>


using namespace casadi;
using namespace std;
namespace fs = std::filesystem;


int main(){
    /*  Test problem
     *
     *    min x0^2 + x1^2
     *    s.t.    x0 + x1 - 10 = 0
     */


    // file name
    std::string file_name = "nlp_code";
    // code predix
    std::string prefix_code = fs::current_path().string() + "/";
    // shared library prefix
    std::string prefix_lib = fs::current_path().string() + "/";

    // Create a new NLP solver instance from the compiled code
    std::string lib_name = prefix_lib + file_name + ".so";
    cout << lib_name <<endl;
    casadi::Function solver = casadi::nlpsol("solver", "ipopt", lib_name);

    //////////////////////////////////////////////////////////////////////////////////////////

    // Number of differential states
    const int numStates = 13;

    // Number of controls
    const int numControls = 6;

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

    double xPos = 1.5;
    double xVel = 0.001;
    double yPos = (2/n)*xVel;
    double yVel = -2*n*xPos;
    double zPos = 1.5;
    double zVel = 0.001;
    std::vector<double> q0 = { 0.771517, 0.46291, 0.308607, 0.308607 }; // Normalized [0.5,0.3,0.2,0.2]
    std::vector<double> wc0 = { 0.0, 0.0, -0.005 };

    // Bounds and initial guess for the state
    std::vector<double> x0_min = {  xPos, yPos, zPos, xVel, yVel, zVel }; // initial position and velocity
    x0_min.insert(x0_min.end(),q0.begin(),q0.end()); // append initial quaternion
    x0_min.insert(x0_min.end(),wc0.begin(),wc0.end()); // append initial angular velocity

    std::vector<double> x0_max = {  xPos, yPos, zPos, xVel, yVel, zVel };
    x0_max.insert(x0_max.end(),q0.begin(),q0.end()); // append initial quaternion
    x0_max.insert(x0_max.end(),wc0.begin(),wc0.end()); // append initial angular velocity

    std::vector<double> x_min  = { -inf, -inf, -inf, -inf, -inf, -inf, -1, -1, -1, -1, -inf, -inf, -inf };
    std::vector<double> x_max  = { inf, inf, inf, inf, inf, inf, 1, 1, 1, 1, inf, inf, inf };
    std::vector<double> xf_min = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> xf_max = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> x_init = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    // MPC Horizon and Sampling Period
    const int N = 1; // Prediction Horizon
    double ts = 3.0; // sampling period
    double tf = ts*N;

    // Total number of NLP variables
    const int numVars = numStates*(N+1) + numControls*N;

    // Declare variable vector for the NLP
    MX V = MX::sym("V",numVars);

    // NLP variable bounds and initial guess
    std::vector<double> v_min,v_max,v_init;

    // Offset in V
    int offset=0;

    // State at each shooting node and control for each shooting interval
    std::vector<MX> X, U;
    for(int k=0; k<N; ++k){
        // Local state
        X.push_back( V.nz(Slice(offset,offset + numStates)));
        if(k==0){
            v_min.insert(v_min.end(), x0_min.begin(), x0_min.end());
            v_max.insert(v_max.end(), x0_max.begin(), x0_max.end());
        } else {
            v_min.insert(v_min.end(), x_min.begin(), x_min.end());
            v_max.insert(v_max.end(), x_max.begin(), x_max.end());
        }
        v_init.insert(v_init.end(), x_init.begin(), x_init.end());
        offset += numStates;

        // Local control
        U.push_back( V.nz(Slice(offset,offset + numControls)));
        v_min.insert(v_min.end(), u_min.begin(), u_min.end());
        v_max.insert(v_max.end(), u_max.begin(), u_max.end());
        v_init.insert(v_init.end(), u_init.begin(), u_init.end());
        offset += numControls;
    }

    // State at end
    X.push_back(V.nz(Slice(offset,offset+numStates)));
    v_min.insert(v_min.end(), xf_min.begin(), xf_min.end());
    v_max.insert(v_max.end(), xf_max.begin(), xf_max.end());
    v_init.insert(v_init.end(), x_init.begin(), x_init.end());
    offset += numStates;

    // Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
    casadi_assert(offset==numVars, "");

    // Bounds and initial guess
    std::map<std::string, casadi::DM> arg, res;
    arg["lbx"] = v_min;
    arg["ubx"] = v_max;
    arg["lbg"] = 0;
    arg["ubg"] = 0;
    arg["x0"] = v_init;

//    arg["lbx"] = -casadi::DM::inf();
//    arg["ubx"] =  casadi::DM::inf();
//    arg["lbg"] =  0;
//    arg["ubg"] =  casadi::DM::inf();
//    arg["x0"] = 0;
//
    // Solve the NLP
    res = solver(arg);
//
//    // Print solution
//    std::cout << "-----" << std::endl;
//    std::cout << "objective at solution = " << res.at("f") << std::endl;
//    std::cout << "primal solution = " << res.at("x") << std::endl;
//    std::cout << "dual solution (x) = " << res.at("lam_x") << std::endl;
//    std::cout << "dual solution (g) = " << res.at("lam_g") << std::endl;

    return 0;
}