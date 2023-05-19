# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

# Often this is installed in /opt/mellanox/libdpdk. If so, run `set(libdpdk_ROOT "/opt/mellanox/libdpdk")` before calling this
# file

# List of required args. Start with INCLUDE_DIR since the first one is displayed

list(APPEND libdpdk_REQUIRED_VARS libdpdk_INCLUDE_DIR)

# Find BSD
find_package(bsd QUIET)
list(APPEND doca_REQUIRED_VARS bsd_FOUND)

# CMAKE_LIBRARY_ARCHITECTURE needs to be set for this to work correctly. Save the value so we can restore
set(_saved_CMAKE_LIBRARY_ARCHITECTURE ${CMAKE_LIBRARY_ARCHITECTURE})

if(NOT DEFINED CMAKE_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE x86_64-linux-gnu)
endif()

# Find the include path
find_path(
  libdpdk_INCLUDE_DIR rte_eal.h
  PATH_SUFFIXES dpdk
)
mark_as_advanced(libdpdk_INCLUDE_DIR)

# Find all of the libraries
list(APPEND libdpdk_REQUIRED_LIBS
  rte_eal
)

foreach(library_name IN LISTS libdpdk_REQUIRED_LIBS)
  find_library(
    ${library_name}_LIBRARY
    NAMES ${library_name}
  )
  mark_as_advanced(${library_name}_LIBRARY)
  list(APPEND libdpdk_REQUIRED_VARS ${library_name}_LIBRARY)
endforeach()

macro(parse_define_number define_name file_string output_variable)
  string(REGEX MATCH "#define ${define_name} ([^\n]+)\n" _ "${file_string}")
  set(${output_variable} "${CMAKE_MATCH_1}")
endmacro()

macro(parse_define_string define_name file_string output_variable)
  string(REGEX MATCH "#define ${define_name} \"([^\n]+)\"\n" _ "${file_string}")
  set(${output_variable} "${CMAKE_MATCH_1}")
endmacro()

# Restore CMAKE_LIBRARY_ARCHITECTURE now that we are done
set(CMAKE_LIBRARY_ARCHITECTURE ${_saved_CMAKE_LIBRARY_ARCHITECTURE})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libdpdk
  FOUND_VAR libdpdk_FOUND
  REQUIRED_VARS
    ${libdpdk_REQUIRED_VARS}
)

if(libdpdk_FOUND)

  list(APPEND libdpdk_child_targets)

  foreach(library_name IN LISTS libdpdk_REQUIRED_LIBS)
    if(NOT TARGET libdpdk::${library_name})
      add_library(${library_name} SHARED IMPORTED GLOBAL)
      add_library(libdpdk::${library_name} ALIAS ${library_name})
      set_target_properties(${library_name} PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${${library_name}_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${libdpdk_INCLUDE_DIR};${libdpdk_INCLUDE_DIR}/../x86_64-linux-gnu/dpdk"
        INTERFACE_LINK_LIBRARIES "bsd::bsd"
      )

      # Add to the list of dependent targets
      list(APPEND libdpdk_child_targets libdpdk::${library_name})
    endif()
  endforeach()

  if(NOT TARGET libdpdk::libdpdk)
    add_library(libdpdk::libdpdk INTERFACE IMPORTED)
    set_target_properties(libdpdk::libdpdk PROPERTIES
      INTERFACE_LINK_LIBRARIES "${libdpdk_child_targets}"
    )
  endif()
endif()
