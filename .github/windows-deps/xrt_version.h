/**
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (C) 2019-2021 Xilinx, Inc. All rights reserved.
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef _XRT_VERSION_H_
#define _XRT_VERSION_H_

static const char xrt_build_version[] = "2.18.0";

static const char xrt_build_version_branch[] = "master";

static const char xrt_build_version_hash[] = "0000000000000000000000000000000000000000";

static const char xrt_build_version_hash_date[] = "unknown";

static const char xrt_build_version_date_rfc[] = "unknown";

static const char xrt_build_version_date[] = "unknown";

static const char xrt_modified_files[] = "";

#define XRT_DRIVER_VERSION "2.18.0,0000000000000000000000000000000000000000"

#define XRT_VERSION(major, minor) ((major << 16) + (minor))
#define XRT_VERSION_CODE XRT_VERSION(2, 18)
#define XRT_MAJOR(code) ((code >> 16))
#define XRT_MINOR(code) (code - ((code >> 16) << 16))
#define XRT_PATCH 0
#define XRT_HEAD_COMMITS 0
#define XRT_BRANCH_COMMITS 0

#ifdef __cplusplus
#include <iostream>
#include <string>

namespace xrt::version {

inline void
print(std::ostream & output)
{
  output << "       XRT Build Version: " << xrt_build_version << std::endl;
  output << "    Build Version Branch: " << xrt_build_version_branch << std::endl;
  output << "      Build Version Hash: " << xrt_build_version_hash << std::endl;
  output << " Build Version Hash Date: " << xrt_build_version_hash_date << std::endl;
  output << "      Build Version Date: " << xrt_build_version_date_rfc << std::endl;
}

} // namespace xrt::version
#endif // __cplusplus

#endif
