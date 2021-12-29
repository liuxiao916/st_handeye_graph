#include <regex>
#include <memory>
#include <random>
#include <sstream>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <st_handeye/st_handeye.hpp>

// cctag
#include "cctag/ICCTag.hpp"
#include "cctag/CCTag.hpp"

using namespace cctag;

/**
 * @brief The Dataset struct
 */
struct Dataset {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // converts Eigen::Matrix to cv::Mat
    static cv::Mat eigen2cv(const Eigen::MatrixXd& mat) {
        cv::Mat cv_mat(mat.rows(), mat.cols(), CV_32FC1);
        for(int i=0; i<mat.rows(); i++) {
            for(int j=0; j<mat.cols(); j++) {
                cv_mat.at<float>(i, j) = mat(i, j);
            }
        }
        return cv_mat;
    }

    // reads a matrix from a csv
    static Eigen::MatrixXd read_matrix(const std::string& filename) {
        std::ifstream ifs(filename);
        if(!ifs) {
            std::cerr << "failed to open " << filename << std::endl;
            return Eigen::MatrixXd();
        }

        std::vector<std::vector<double>> matrix;
        while(!ifs.eof()) {
            std::string line;
            std::getline(ifs, line);

            if(line.empty()) {
                continue;
            }

            std::vector<double> row;
            std::stringstream sst(line);
            int i = 0;
            while(!sst.eof()) {
                double value;
                sst >> value;
                if(i<4){
                    row.push_back(value);
                    i++;
                }
            }

            matrix.push_back(row);
        }


        Eigen::MatrixXd m(matrix.size(), matrix[0].size());
        for(int i=0; i<matrix.size(); i++) {
            for(int j=0; j<matrix[i].size(); j++) {
                m(i, j) = matrix[i][j];
            }
        }
        return m;
    }

    // reads data from a directory
    static std::shared_ptr<Dataset> read(const std::string& dataset_dir, const std::string& ros_camera_params_file, bool visualize) {
        int PATTERN_ROWS = 3;
        int PATTERN_COLS = 4;
        double L = 0.065;
        std::shared_ptr<Dataset> dataset(new Dataset());
        dataset->pattern_3d.resize(3, PATTERN_ROWS * PATTERN_COLS);
        for(int j=0; j<PATTERN_COLS; j++) {
            for(int i=0; i<PATTERN_ROWS; i++) {
                int idx = j * PATTERN_ROWS + i;
                dataset->pattern_3d.col(idx) = Eigen::Vector3d(-j*L ,i*L , 0);
            }
        }

        // for(int j=0; j<PATTERN_COLS; j++) {
        //     for(int i=0; i<PATTERN_ROWS; i++) {
        //         double y_offset = j % 2 == 0 ? 0 : L / 2;
        //         int idx = j * PATTERN_ROWS + i;
        //         double X = j * L / 2;
        //         double Y = i * L + y_offset;
        //         //dataset->pattern_3d.col(idx) = Eigen::Vector3d(j * L / 2, i * L + y_offset, 0);
        //     }
        // }

        // for(int j=0; j<PATTERN_COLS; j++) {
        //     for(int i=0; i<PATTERN_ROWS; i++) {
        //         double y_offset = j % 2 == 0 ? 0 : L / 2;
        //         int idx = j * PATTERN_ROWS + i;
        //         std::cout<<dataset->pattern_3d.col(idx)<<std::endl;
        //     }
        //     std::cout<<std::endl;
        // }

        if(!ros_camera_params_file.empty()) {
            if(!dataset->read_ros_camera_params(ros_camera_params_file)) {
                return nullptr;
            }
        }

        boost::filesystem::directory_iterator dir(dataset_dir);
        boost::filesystem::directory_iterator end;

        for(dir; dir != end; dir++) {
            std::string filename = dir->path().string();

            auto suffix_loc = filename.find("_image.jpg");
            if(suffix_loc == std::string::npos) {
                continue;
            }

            std::string data_id = filename.substr(suffix_loc - 3, 3);

            cv::Mat image = cv::imread(dataset_dir + "/" + data_id + "_image.jpg");
            cv::cvtColor(image, image, CV_BGR2GRAY);
            Eigen::Matrix4d handpose = read_matrix(dataset_dir + "/" + data_id + "_pose.csv");

            if(ros_camera_params_file.empty()) {
                dataset->camera_matrix = read_matrix(dataset_dir + "/" + data_id + "_camera_matrix.csv");
                dataset->distortion = read_matrix(dataset_dir + "/" + data_id + "_distortion.csv");
            }

            cv::Mat cv_camera_matrix = eigen2cv(dataset->camera_matrix);
            cv::Mat cv_distortion = eigen2cv(dataset->distortion);

            cv::Mat undistorted;
            cv::undistort(image, undistorted, cv_camera_matrix, cv_distortion);

            //detect cctag
            size_t frameId = 0;
            int pipeId = 0;
            boost::ptr_list<cctag::ICCTag> markers{};
            const size_t nCrowns = 3;
            Parameters params(nCrowns);
            params.setUseCuda(false);
            cctagDetection(markers, pipeId, frameId, undistorted, params);
            int i = 0;
            Eigen::MatrixXd grid_2d(2, PATTERN_ROWS * PATTERN_COLS);
            for(const cctag::ICCTag& marker : markers)
            {
                //std::cout<<i<<std::endl;
                if(marker.getStatus() == status::id_reliable){
                    grid_2d(0, i) = marker.x();
                    grid_2d(1, i) = marker.y();                 
                    i++;
                }
            }


            // cv::Mat cv_grid_2d;
            // bool ret = cv::findCirclesGrid(undistorted, cv::Size(PATTERN_ROWS, PATTERN_COLS), cv_grid_2d, cv::CALIB_CB_ASYMMETRIC_GRID);
            // cv::drawChessboardCorners(undistorted, cv::Size(PATTERN_ROWS, PATTERN_COLS), cv_grid_2d, ret);
            //imwrite(data_id+".jpg", undistorted);
            // if(!ret) {
            //     std::cerr << "failed to find circles!!" << std::endl;
            //     std::cerr << filename << std::endl;
            //     continue;
            // }

            // Eigen::MatrixXd grid_2d(2, PATTERN_ROWS * PATTERN_COLS);
            // for(int i=0; i<12; i++) {
            //     grid_2d(0, i) = cv_grid_2d.at<cv::Vec2f>(i)[0];
            //     grid_2d(1, i) = cv_grid_2d.at<cv::Vec2f>(i)[1];
            // }

            // for(int i=0; i<12; i++) {
            //     std::cout<< grid_2d(0, i) << " "<<grid_2d(1, i)<<std::endl;
            // }

            //std::cout<<std::endl;
            dataset->images.push_back(undistorted);
            dataset->handposes.push_back(handpose);
            dataset->pattern_2ds.push_back(grid_2d);

            if(visualize) {
                std::cout << "--- handpose ---\n" << handpose << std::endl;
                std::cout << "--- grid_2d ---\n" << grid_2d.transpose() << std::endl;

                cv::Mat resized;
                cv::resize(undistorted, resized, cv::Size(undistorted.cols / 4, undistorted.rows / 4));

                cv::imshow("undistorted", resized);
                cv::waitKey(100);
            }
        }


        std::cout << "num_images: " << dataset->images.size() << std::endl;

        return dataset;
    }

    Eigen::MatrixXd read_matrix_from_yaml(std::ifstream& ifs) {
        std::string token;
        int rows, cols;
        ifs >> token >> rows >> token >> cols;

        Eigen::MatrixXd matrix(rows, cols);

        char c;
        ifs >> token >> c;

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                ifs >> matrix(i, j) >> c;
            }
        }

        return matrix;
    }

    bool read_ros_camera_params(const std::string& ros_camera_params_file) {
        std::ifstream ifs(ros_camera_params_file);
        if(!ifs) {
            std::cerr << "failed to open the file!!" << std::endl;
            std::cerr << ros_camera_params_file;
            return false;
        }

        std::string line;
        while(!ifs.eof() && line.find("camera_matrix") == std::string::npos) {
            std::getline(ifs, line);
        }
        camera_matrix = read_matrix_from_yaml(ifs);
        std::cout << "--- camera_matrix ---\n" << camera_matrix << std::endl;

        while(!ifs.eof() && line.find("distortion_coefficients") == std::string::npos) {
            std::getline(ifs, line);
        }
        distortion = read_matrix_from_yaml(ifs).transpose();
        std::cout << "--- distortion ---\n" << distortion << std::endl;

        return true;
    }

public:
    Eigen::Matrix3d camera_matrix;
    Eigen::VectorXd distortion;

    Eigen::MatrixXd pattern_3d;
    std::vector<cv::Mat> images;
    std::vector<Eigen::MatrixXd> pattern_2ds;
    std::vector<Eigen::Matrix4d> handposes;
};


int main(int argc, char** argv) {
    using namespace boost::program_options;

    options_description description("calibrate");
    description.add_options()
            ("visualize,v", "if visualize")
            ("use_init_guess,u", "if true, the tsai's result is given to the graph-based method as initial guess")
            ("visual_inf_scale", value<double>()->default_value(1.0), "scale of the informatoin matrix for visual detections")
            ("handpose_inf_scale_trans", value<double>()->default_value(1.0), "scale of the informatoin matrix for hand poses")
            ("handpose_inf_scale_rot", value<double>()->default_value(1.0), "scale of the informatoin matrix for hand poses")
            ("num_iterations", value<int>()->default_value(8192), "max number of iterations")
            ("solver_name", value<std::string>()->default_value("lm_var_cholmod"), "g2o solver name")
            ("robust_kernel_handpose", value<std::string>()->default_value("NONE"), "robust kernel for handpose edges")
            ("robust_kernel_projection", value<std::string>()->default_value("NONE"), "robust kernel for projection edges")
            ("robust_kernel_handpose_delta", value<double>()->default_value(1.0), "robust kernel delta for handpose edges")
            ("robust_kernel_projection_delta", value<double>()->default_value(1.0), "robust kernel delta for projection edges")
            ("directory", value<std::string>(), "input directory")
            ("camera_params,c", value<std::string>(), "if true, substitute an external ROS format camera matrix file for the default ones")
            ("save_hand2eye_visp", value<std::string>(), "file where the estimated hand2eye transformation will be written")
            ("save_hand2eye_dq", value<std::string>(), "file where the estimated hand2eye transformation will be written")
            ("save_hand2eye_graph", value<std::string>(), "file where the estimated hand2eye transformation will be written")
    ;

    variables_map vm;
    store(command_line_parser{argc, argv}
        .options(description)
        .positional(
            positional_options_description {}
            .add("directory", 1)
        ).run()
    , vm);

    notify(vm);

    std::cout << "read images..." << std::endl;
    std::string dataset_dir = vm["directory"].as<std::string>();

    std::string ros_camera_params_file;
    if(vm.count("camera_params")) {
        ros_camera_params_file = vm["camera_params"].as<std::string>();
    }

    auto dataset = Dataset::read(dataset_dir, ros_camera_params_file, vm.count("visualize"));

    std::vector<Eigen::Isometry3d> world2hands(dataset->handposes.size());
    for(int i=0; i<dataset->handposes.size(); i++) {
        world2hands[i] = Eigen::Isometry3d(dataset->handposes[i]).inverse();
    }

    // optimization params
    st_handeye::OptimizationParams params;
    params.pattern2d_inf_scale = vm["visual_inf_scale"].as<double>();
    params.world2hand_inf_scale_trans = vm["handpose_inf_scale_trans"].as<double>();
    params.world2hand_inf_scale_rot = vm["handpose_inf_scale_rot"].as<double>();
    params.num_iterations = vm["num_iterations"].as<int>();
    params.solver_name = vm["solver_name"].as<std::string>();
    params.robust_kernel_handpose = vm["robust_kernel_handpose"].as<std::string>();
    params.robust_kernel_projection = vm["robust_kernel_projection"].as<std::string>();
    params.robust_kernel_handpose_delta = vm["robust_kernel_handpose_delta"].as<double>();
    params.robust_kernel_projection_delta = vm["robust_kernel_projection_delta"].as<double>();

    std::cout << "calibrate_visp..." << std::endl;
    Eigen::Isometry3d hand2eye_visp = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d object2world_visp = Eigen::Isometry3d::Identity();
    st_handeye::spatial_calibration_visp(dataset->camera_matrix, dataset->pattern_3d, world2hands, dataset->pattern_2ds, hand2eye_visp, object2world_visp, params);

    std::cout << "calibrate_dq..." << std::endl;
    Eigen::Isometry3d hand2eye_dq = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d object2world_dq = Eigen::Isometry3d::Identity();
    st_handeye::spatial_calibration_dualquaternion(dataset->camera_matrix, dataset->pattern_3d, world2hands, dataset->pattern_2ds, hand2eye_dq, object2world_dq, params);

    std::cout << "calibrate_graph..." << std::endl;
    Eigen::Isometry3d hand2eye_graph = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d object2world_graph = Eigen::Isometry3d::Identity();


    Eigen::Isometry3d init_guess = Eigen::Isometry3d::Identity();
    init_guess(0,3)=0.05;
    init_guess(2,3)=0.2;
    std::cout<<"init"<<std::endl;
    std::cout<<init_guess.matrix()<<std::endl;
    if(vm.count("use_init_guess")) {
        std::cout<<"USE"<<std::endl;
        //hand2eye_graph = hand2eye_visp;
        hand2eye_graph = init_guess;
        object2world_graph = object2world_visp;
    }

    st_handeye::spatial_calibration_graph(dataset->camera_matrix, dataset->pattern_3d, world2hands, dataset->pattern_2ds, hand2eye_graph, object2world_graph, params);

    std::cout << "--- hand2eye_visp ---\n" << hand2eye_visp.matrix().inverse() << std::endl;
    std::cout << "--- hand2eye_dq ---\n" << hand2eye_dq.matrix().inverse() << std::endl;
    std::cout << "--- hand2eye_graph ---\n" << hand2eye_graph.matrix().inverse() << std::endl;
    std::cout << "--- object2world_visp ---\n" << object2world_visp.matrix().inverse() << std::endl;
    std::cout << "--- object2world_dq ---\n" << object2world_dq.matrix().inverse() << std::endl;
    std::cout << "--- object2world_graph ---\n" << object2world_graph.matrix().inverse() << std::endl;

    if(vm.count("save_hand2eye_visp")) {
        std::ofstream ofs(vm["save_hand2eye_visp"].as<std::string>());
        ofs << hand2eye_visp.matrix() << std::endl;
    }
    if(vm.count("save_hand2eye_dq")) {
        std::ofstream ofs(vm["save_hand2eye_dq"].as<std::string>());
        ofs << hand2eye_dq.matrix() << std::endl;
    }
    if(vm.count("save_hand2eye_graph")) {
        std::ofstream ofs(vm["save_hand2eye_graph"].as<std::string>());
        ofs << hand2eye_graph.matrix() << std::endl;
    }

    return 0;
}
