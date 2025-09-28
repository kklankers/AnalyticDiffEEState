#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Dense>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/math/quaternion.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/spatial/skew.hpp>

// Forward declarations for the functions
Eigen::Matrix<double, 19, 1> compute_cartesian_state(
    const pinocchio::Model& model, pinocchio::FrameIndex frame_id,
    const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& ddq);

Eigen::MatrixXd compute_ee_state_jacobian_analytical(
    const pinocchio::Model& model, pinocchio::FrameIndex frame_id,
    const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& ddq);

Eigen::MatrixXd compute_ee_state_jacobian_numerical(
    const pinocchio::Model& model, pinocchio::FrameIndex frame_id,
    const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& ddq,
    double eps);

/**
 * @brief Computes the 19-element cartesian state of a frame.
 */
Eigen::Matrix<double, 19, 1> compute_cartesian_state(
    const pinocchio::Model& model, pinocchio::FrameIndex frame_id,
    const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& ddq)
{
    pinocchio::Data data = pinocchio::Data(model);

    pinocchio::forwardKinematics(model, data, q, dq, ddq);
    pinocchio::updateFramePlacements(model, data);

    const pinocchio::SE3& frame_pose = data.oMf[frame_id];
    const auto& position = frame_pose.translation();
    const Eigen::Quaternion<double> quaternion(frame_pose.rotation());

    const pinocchio::Motion velocity = pinocchio::getFrameVelocity(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    const pinocchio::Motion acceleration = pinocchio::getFrameAcceleration(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED);

    Eigen::Matrix<double, 19, 1> cartesian_state;
    cartesian_state << position,
                       quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w(),
                       velocity.linear(),
                       velocity.angular(),
                       acceleration.linear(),
                       acceleration.angular();
    return cartesian_state;
}

/**
 * @brief Computes the analytical Jacobian of the end-effector state.
 */
Eigen::MatrixXd compute_ee_state_jacobian_analytical(
    const pinocchio::Model& model, pinocchio::FrameIndex frame_id,
    const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& ddq)
{
    pinocchio::Data data = pinocchio::Data(model);
    const int nv = model.nv;
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(19, 3 * nv);

    pinocchio::computeForwardKinematicsDerivatives(model, data, q, dq, ddq);
    pinocchio::updateFramePlacements(model, data);
    Eigen::MatrixXd dv_dq(6, nv), dv_dv(6, nv), da_dq(6, nv), da_dv(6, nv), da_da(6, nv);
    pinocchio::getFrameAccelerationDerivatives(
        model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED,
        dv_dq, dv_dv, da_dq, da_dv, da_da
    );

    const Eigen::MatrixXd& J = dv_dv; // This is the geometric Jacobian in world frame
    const pinocchio::Motion frame_vel = pinocchio::getFrameVelocity(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    const Eigen::Vector3d& v = frame_vel.linear();
    const Eigen::Vector3d& w = frame_vel.angular();

    // The classical velocity derivative requires a correction term
    Eigen::MatrixXd dvc_dq = dv_dq.topRows(3) + pinocchio::skew(w) * J.topRows(3);
    const Eigen::Ref<const Eigen::MatrixXd> dw_dq = dv_dq.bottomRows(3);

    // The classical acceleration derivative requires a more complex correction term
    Eigen::MatrixXd d_wxv_dq = Eigen::MatrixXd::Zero(3, nv);
    for (int i = 0; i < nv; ++i) {
        Eigen::Vector3d dw_dq_i = dw_dq.col(i);
        d_wxv_dq.col(i) = pinocchio::skew(dw_dq_i) * v + pinocchio::skew(w) * dvc_dq.col(i);
    }
    Eigen::MatrixXd dac_dq = da_dq.topRows(3) - d_wxv_dq;

    // --- Assemble the full Jacobian ---
    // Partials with respect to q
    jacobian.block(0, 0, 3, nv) = J.topRows<3>();
    const pinocchio::SE3& ee_pose = data.oMf[frame_id];
    Eigen::Quaterniond q_current(ee_pose.rotation());
    Eigen::Matrix<double, 4, 3> E_quat;
    E_quat <<  q_current.w(),  q_current.z(), -q_current.y(),
              -q_current.z(),  q_current.w(),  q_current.x(),
               q_current.y(), -q_current.x(),  q_current.w(),
              -q_current.x(), -q_current.y(), -q_current.z();
    jacobian.block(3, 0, 4, nv) = 0.5 * E_quat * J.bottomRows<3>();
    jacobian.block(7, 0, 3, nv) = dvc_dq;
    jacobian.block(10, 0, 3, nv) = dw_dq;
    jacobian.block(13, 0, 3, nv) = dac_dq;
    jacobian.block(16, 0, 3, nv) = da_dq.bottomRows(3);

    // Partials with respect to dq
    jacobian.block(7, nv, 6, nv) = J;
    jacobian.block(13, nv, 6, nv) = da_dv;

    // Partials with respect to ddq
    jacobian.block(13, 2 * nv, 6, nv) = da_da;

    return jacobian;
}

/**
 * @brief Numerically computes the Jacobian of the end-effector state using finite differences.
 * This serves as a ground truth to verify the analytical version.
 */
Eigen::MatrixXd compute_ee_state_jacobian_numerical(
    const pinocchio::Model& model, pinocchio::FrameIndex frame_id,
    const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& ddq,
    double eps = 1e-7)
{
    const int nv = q.size();
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(19, 3 * nv);
    Eigen::Matrix<double, 19, 1> ee_state_base = compute_cartesian_state(model, frame_id, q, dq, ddq);

    // Perturb q
    for (int i = 0; i < nv; ++i) {
        Eigen::VectorXd q_plus = q;
        q_plus(i) += eps;
        jacobian.col(i) = (compute_cartesian_state(model, frame_id, q_plus, dq, ddq) - ee_state_base) / eps;
    }
    // Perturb dq
    for (int i = 0; i < nv; ++i) {
        Eigen::VectorXd dq_plus = dq;
        dq_plus(i) += eps;
        jacobian.col(nv + i) = (compute_cartesian_state(model, frame_id, q, dq_plus, ddq) - ee_state_base) / eps;
    }
    // Perturb ddq
    for (int i = 0; i < nv; ++i) {
        Eigen::VectorXd ddq_plus = ddq;
        ddq_plus(i) += eps;
        jacobian.col(2 * nv + i) = (compute_cartesian_state(model, frame_id, q, dq, ddq_plus) - ee_state_base) / eps;
    }

    return jacobian;
}

int main() {
    // --- Setup ---
    pinocchio::Model model;
    // std::string urdf_path = "define absolut path if needed/franka_model.urdf";

    std::string urdf_path = "../franka_model.urdf"; // path if build with CLION
    pinocchio::urdf::buildModel(urdf_path, model);

    const std::string frame_name = "tool_attachment";
    if (!model.existFrame(frame_name)) {
        throw std::runtime_error("Frame '" + frame_name + "' does not exist in the model.");
    }
    const pinocchio::FrameIndex frame_id = model.getFrameId(frame_name);
    const int nq = model.nv;

    // --- Define a test state ---
    Eigen::VectorXd q_test = Eigen::VectorXd::Ones(nq)*0.2;
    Eigen::VectorXd dq_test = Eigen::VectorXd::Ones(nq) * 1.3;
    Eigen::VectorXd ddq_test = Eigen::VectorXd::Ones(nq) * 0.5;

    std::cout << "--- Comparing Analytical and Numerical Jacobians ---" << std::endl;
    std::cout << "Number of joints: " << nq << std::endl;
    std::cout << "q_test:   " << q_test.transpose() << std::endl;
    std::cout << "dq_test:  " << dq_test.transpose() << std::endl;
    std::cout << "ddq_test: " << ddq_test.transpose() << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    // --- Compute Jacobians ---
    Eigen::MatrixXd J_analytical = compute_ee_state_jacobian_analytical(model, frame_id, q_test, dq_test, ddq_test);
    Eigen::MatrixXd J_numerical = compute_ee_state_jacobian_numerical(model, frame_id, q_test, dq_test, ddq_test);
    Eigen::MatrixXd J_error = J_analytical - J_numerical;

    // --- Print Results ---
    // Set Eigen print options for better readability
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

    std::cout << "\nANALYTICAL JACOBIAN (19 x " << 3 * nq << "):\n" << J_analytical << std::endl;
    std::cout << "\n----------------------------------------------------\n" << std::endl;
    std::cout << "NUMERICAL JACOBIAN (19 x " << 3 * nq << "):\n" << J_numerical<< std::endl;
    std::cout << "\n----------------------------------------------------\n" << std::endl;

    // Filter out very small values for cleaner error printing
    auto formatted_error = J_error.unaryExpr([](double v) {
        return std::abs(v) < 1e-6 ? 0.0 : v;
    });

    std::cout << "ERROR (Analytical - Numerical):\n" << formatted_error << std::endl;
    std::cout << "\n----------------------------------------------------\n" << std::endl;
    std::cout << "Max absolute error: " << J_error.cwiseAbs().maxCoeff() << std::endl;

    return 0;
}

