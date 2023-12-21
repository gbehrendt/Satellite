
#include <iostream>
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <fstream>

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

int main() {






















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

//    MatrixXd test(13,1), dummy(6,1);
//    test << 0,0,0,0,0,0,1,0,0,0,0,0,0 ;
//    dummy << 0,0,0,0,0,0;
//    cout << f(test,dummy,n,mc) << endl;

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



