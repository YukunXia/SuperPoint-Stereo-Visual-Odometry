#pragma once

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

struct CostFunctor32 { // 32 means 3d - 2d observation pair
  CostFunctor32(const cv::Vec3f &point_3d, const cv::Point2f &point_2d,
                const cv::Mat &projection, const bool inverse_transformation)
      : // 3d point on current frame
        point_3d_(point_3d[0], point_3d[1], point_3d[2]),
        // 2d point on current frame
        point_2d_(point_2d.x, point_2d.y),
        // cautious: projection has to be float typed
        projection_((Eigen::Matrix<double, 3, 4, Eigen::RowMajor>()
                         << projection.at<double>(0, 0),
                     projection.at<double>(0, 1), projection.at<double>(0, 2),
                     projection.at<double>(0, 3), projection.at<double>(1, 0),
                     projection.at<double>(1, 1), projection.at<double>(1, 2),
                     projection.at<double>(1, 3), projection.at<double>(2, 0),
                     projection.at<double>(2, 1), projection.at<double>(2, 2),
                     projection.at<double>(2, 3))
                        .finished()),
        inverse_transformation_(inverse_transformation) {}

  template <typename T>
  bool operator()(const T q[4], const T t[3], T residuals[2]) const {
    // take in variables to be optimized
    const Eigen::Map<const Eigen::Quaternion<T>> q_(q);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_(t);
    Eigen::Transform<T, 3, Eigen::Isometry> transformation;
    transformation.linear() = q_.toRotationMatrix();
    transformation.translation() = t_;
    if (inverse_transformation_) {
      transformation = transformation.inverse();
    }

    // 3d point on curr frame
    const Eigen::Matrix<T, 3, 1> point_3d_T = point_3d_.template cast<T>();
    // 2d point on prev frame
    const Eigen::Matrix<T, 2, 1> point_2d_T = point_2d_.template cast<T>();
    const Eigen::Matrix<T, 3, 4, Eigen::RowMajor> projection_T =
        projection_.template cast<T>();

    // \tilde{p_2d} = [K | baseline] * [R / 0 | T / 1] * \tilde{P_3d}
    const Eigen::Matrix<T, 3, 1> point_3d_T_transformed =
        transformation * point_3d_T.homogeneous();
    const Eigen::Matrix<T, 3, 1> point_3d_T_projected =
        projection_T * point_3d_T_transformed.homogeneous();

    residuals[0] = point_3d_T_projected(0, 0) / point_3d_T_projected(2, 0) -
                   point_2d_T(0, 0);
    residuals[1] = point_3d_T_projected(1, 0) / point_3d_T_projected(2, 0) -
                   point_2d_T(1, 0);

    return true;
  }

  static ceres::CostFunction *Create(const cv::Vec3d &point_3d,
                                     const cv::Point2f &point_2d,
                                     const cv::Mat &projection,
                                     const bool inverse_transformation) {
    return (new ceres::AutoDiffCostFunction<CostFunctor32, 2, 4, 3>(
        new CostFunctor32(point_3d, point_2d, projection,
                          inverse_transformation)));
  }

  const Eigen::Vector3d point_3d_;
  const Eigen::Vector2d point_2d_;
  const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> projection_;
  // const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
  //                                      Eigen::RowMajor>>
  //     projection_;
  const bool inverse_transformation_;
};