/// \file vm_args.hpp
/// \brief vm_args class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note This class is used to parse the command line arguments.
#pragma once

#include <boost/program_options.hpp>
#include <iostream>

namespace arg_utils {

namespace po = boost::program_options;

/// \brief add the default options
/// \param desc the options description
void add_default_options(po::options_description &desc) {
    desc.add_options()("help,h", "produce help message");
    desc.add_options()("device,d", po::value<std::string>()->default_value("npu1"), "Device type, npu1 or npu2");
}

/// \brief parse the options
/// \param argc the number of arguments
/// \param argv the arguments
/// \param desc the options description
/// \param vm the variables map
void parse_options(int argc, char *argv[], po::options_description &desc,
    po::variables_map &vm) {
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            std::exit(1);
        }
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        std::cerr << "Usage:\n" << desc << std::endl;
        std::exit(1);
    }
}


}
