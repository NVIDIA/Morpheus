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

# Often this is installed in /opt/mellanox/dpdk. If so, run `set(libdpdk_ROOT "/opt/mellanox/dpdk")` before calling this
# file

# Find BSD
find_package(bsd)

# Create a scope to allow variables to be set locally without altering the parent scope
block(SCOPE_FOR VARIABLES
  PROPAGATE
    libdpdk_FOUND
    libdpdk_VERSION
)

  # List of required args. Start with INCLUDE_DIR since the first one is displayed
  list(APPEND libdpdk_REQUIRED_VARS bsd_FOUND libdpdk_INCLUDE_DIR)

  # This library will always be installed on the host. Allow that to be searched here (Should fix this up in the future)
  set("CMAKE_FIND_ROOT_PATH_MODE_INCLUDE" BOTH)
  set("CMAKE_FIND_ROOT_PATH_MODE_LIBRARY" BOTH)

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
          INTERFACE_INCLUDE_DIRECTORIES "${libdpdk_INCLUDE_DIR};${libdpdk_INCLUDE_DIR}/../${CMAKE_LIBRARY_ARCHITECTURE}/dpdk"
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
endblock()
