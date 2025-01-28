# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_guard()

function(morpheus_configure_libbsd)
  list(APPEND CMAKE_MESSAGE_CONTEXT "libbsd")

  set(MORPHEUS_LIBBSD_VERSION "0.11.7" CACHE STRING "Version of libbsd to build")

  # Check if bsd is available -- download if not
  rapids_cpm_find(bsd ${MORPHEUS_LIBBSD_VERSION}
    GLOBAL_TARGETS bsd
    CPM_ARGS
      GIT_REPOSITORY          https://gitlab.freedesktop.org/libbsd/libbsd
      GIT_TAG                 ${MORPHEUS_LIBBSD_VERSION}
      DOWNLOAD_ONLY           TRUE
  )

  if (bsd_ADDED)
    message(STATUS "libbsd was not installed and will be built from source")

    find_package(md REQUIRED)

    set(bsd_INSTALL_DIR ${bsd_BINARY_DIR}/install)
    file(MAKE_DIRECTORY ${bsd_INSTALL_DIR}/include)
    include(ExternalProject)

    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE_UC)

    # Get libmd components
    get_target_property(MD_INCLUDE_DIRS md::md INTERFACE_INCLUDE_DIRECTORIES) # Add '-I' to CPPFLAGS
    get_target_property(MD_LIBRARY md::md IMPORTED_LOCATION) # Add '-L' to LDFLAGS

    cmake_path(GET MD_LIBRARY PARENT_PATH MD_LINK_DIRECTORY)

    # Get the Compiler settings to forward onto autoconf
    set(COMPILER_SETTINGS
      "CXX=${CMAKE_CXX_COMPILER_LAUNCHER} ${CMAKE_CXX_COMPILER}"
      "CPP=${CMAKE_CXX_COMPILER_LAUNCHER} ${CMAKE_C_COMPILER} -E"
      "CC=${CMAKE_C_COMPILER_LAUNCHER} ${CMAKE_C_COMPILER}"
      "AR=${CMAKE_C_COMPILER_AR}"
      "RANLIB=${CMAKE_C_COMPILER_RANLIB}"
      "NM=${CMAKE_NM}"
      "STRIP=${CMAKE_STRIP}"
      "CFLAGS=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE_UC}}"
      "CPPFLAGS=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE_UC}} -I${MD_INCLUDE_DIRS}"
      "CXXFLAGS=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UC}}"
      "LDFLAGS=${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${BUILD_TYPE_UC}} -L${MD_LINK_DIRECTORY}"
    )

    ExternalProject_Add(bsd
      PREFIX              ${bsd_BINARY_DIR} # Root directory for bsd
      SOURCE_DIR          ${bsd_BINARY_DIR} # Move source over from cpm dir and build in binary dir
      INSTALL_DIR         ${bsd_INSTALL_DIR}

      DOWNLOAD_COMMAND    ${CMAKE_COMMAND} -E copy_directory ${bsd_SOURCE_DIR} ${bsd_BINARY_DIR}
      # Note, we set SED and GREP here since they can be hard coded in the conda libtoolize
      CONFIGURE_COMMAND   ${CMAKE_COMMAND} -E env SED=sed GREP=grep <SOURCE_DIR>/autogen
      COMMAND   <SOURCE_DIR>/configure ${COMPILER_SETTINGS} --prefix=${CMAKE_INSTALL_PREFIX} --enable-static

      BUILD_COMMAND       make -j
      BUILD_IN_SOURCE     TRUE
      BUILD_BYPRODUCTS    <INSTALL_DIR>/lib/libbsd.a
      INSTALL_COMMAND     make install prefix=<INSTALL_DIR>
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      # Add a target for configuring to allow for style checks on source code
      STEP_TARGETS        install
      DEPENDS             md::md
    )

    # Install only the headers
    install(
      DIRECTORY ${bsd_INSTALL_DIR}/include
      TYPE INCLUDE
    )

    add_library(bsd::bsd STATIC IMPORTED GLOBAL)
    set_target_properties(bsd::bsd
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
          "$<BUILD_INTERFACE:${bsd_INSTALL_DIR}/include>;$<INSTALL_INTERFACE:include>"
        INTERFACE_LINK_LIBRARIES
          "md::md"
        INTERFACE_POSITION_INDEPENDENT_CODE
          "ON"
        IMPORTED_LOCATION
          "${bsd_INSTALL_DIR}/lib/libbsd.a"
        IMPORTED_SONAME
          "libbsd.a"
    )

    add_dependencies(bsd::bsd bsd)

  endif()

  LIST(POP_BACK CMAKE_MESSAGE_CONTEXT)

endfunction()
