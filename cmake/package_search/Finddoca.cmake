# =============================================================================
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Often this is installed in /opt/mellanox/doca. If so, run `set(doca_ROOT "/opt/mellanox/doca")` before calling this
# file

if(DEFINED doca_ROOT)
  # Usually, dpdk is up and over one
  # /opt/mellanox/doca
  # /opt/mellanox/dpdk
  cmake_path(GET doca_ROOT PARENT_PATH mellanox_ROOT)

  set(libdpdk_ROOT "${mellanox_ROOT}/dpdk")
  message(VERBOSE "Setting DPDK search path to match DOCA. libdpdk_ROOT: ${libdpdk_ROOT}")
endif()

# Now find DPDK
find_package(libdpdk)

# Create a scope to allow variables to be set locally without altering the parent scope
block(SCOPE_FOR VARIABLES
  PROPAGATE
    doca_FOUND
    doca_VERSION
)

  # List of required args. Start with INCLUDE_DIR since the first one is displayed
  list(APPEND doca_REQUIRED_VARS libdpdk_FOUND doca_INCLUDE_DIR)

  # This library will always be installed on the host. Allow that to be searched here (Should fix this up in the future)
  set("CMAKE_FIND_ROOT_PATH_MODE_INCLUDE" BOTH)
  set("CMAKE_FIND_ROOT_PATH_MODE_LIBRARY" BOTH)

  # Find the include path
  find_path(
    doca_INCLUDE_DIR doca_gpunetio.h
  )
  mark_as_advanced(doca_INCLUDE_DIR)

  # Find all of the libraries
  list(APPEND doca_REQUIRED_LIBS
    doca_eth
    doca_flow
    doca_gpunetio
    doca_gpunetio_device
  )

  foreach(library_name IN LISTS doca_REQUIRED_LIBS)
    find_library(
      ${library_name}_LIBRARY
      NAMES ${library_name}
    )
    mark_as_advanced(${library_name}_LIBRARY)
    list(APPEND doca_REQUIRED_VARS ${library_name}_LIBRARY)
  endforeach()

  macro(parse_define_number define_name file_string output_variable)
    string(REGEX MATCH "#define ${define_name} ([^\n]+)\n" _ "${file_string}")
    set(${output_variable} "${CMAKE_MATCH_1}")
  endmacro()

  macro(parse_define_string define_name file_string output_variable)
    string(REGEX MATCH "#define ${define_name} \"([^\n]+)\"\n" _ "${file_string}")
    set(${output_variable} "${CMAKE_MATCH_1}")
  endmacro()

  # Parse the version number
  if (DEFINED doca_INCLUDE_DIR)

    find_file(doca_VERSION_FILE
      NAMES doca_version.h
      PATHS ${doca_INCLUDE_DIR}
      NO_DEFAULT_PATH
    )
    mark_as_advanced(doca_VERSION_FILE)

    if (DEFINED doca_VERSION_FILE)

      file(READ ${doca_VERSION_FILE} version_file_string)

      parse_define_string(DOCA_VER_STRING "${version_file_string}" doca_VERSION_FULL)
      parse_define_number(DOCA_VER_MAJOR "${version_file_string}" doca_VERSION_MAJOR)
      parse_define_number(DOCA_VER_MINOR "${version_file_string}" doca_VERSION_MINOR)
      parse_define_number(DOCA_VER_PATCH "${version_file_string}" doca_VERSION_PATCH)

      # Set the version variable
      set(doca_VERSION "${doca_VERSION_FULL}")

    endif()

  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(doca
    FOUND_VAR doca_FOUND
    REQUIRED_VARS
      ${doca_REQUIRED_VARS}
    VERSION_VAR doca_VERSION
  )

  if(doca_FOUND)

    list(APPEND doca_child_targets)

    foreach(library_name IN LISTS doca_REQUIRED_LIBS)
      if(NOT TARGET doca::${library_name})
        add_library(${library_name} UNKNOWN IMPORTED GLOBAL)
        add_library(doca::${library_name} ALIAS ${library_name})
        set_target_properties(${library_name} PROPERTIES
          IMPORTED_LINK_INTERFACE_LANGUAGES "C"
          IMPORTED_LOCATION "${${library_name}_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${doca_INCLUDE_DIR}"
          INTERFACE_LINK_LIBRARIES "libdpdk::libdpdk"
        )

        # Add to the list of dependent targets
        list(APPEND doca_child_targets doca::${library_name})
      endif()
    endforeach()

    if(NOT TARGET doca::doca)
      add_library(doca::doca INTERFACE IMPORTED GLOBAL)
      set_target_properties(doca::doca PROPERTIES
        INTERFACE_LINK_LIBRARIES "${doca_child_targets}"
      )
    endif()
  endif()

endblock()
