# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import gc
import importlib
import logging
import os
import signal
import subprocess
import sys
import time
import types
import typing
import warnings
from unittest import mock

import pytest
import requests

from _utils import import_or_skip
from _utils.kafka import _init_pytest_kafka
from _utils.kafka import kafka_bootstrap_servers_fixture  # noqa: F401 pylint:disable=unused-import
from _utils.kafka import kafka_consumer_fixture  # noqa: F401 pylint:disable=unused-import
from _utils.kafka import kafka_topics_fixture  # noqa: F401 pylint:disable=unused-import

# Don't let pylint complain about pytest fixtures
# pylint: disable=redefined-outer-name,unused-argument

(PYTEST_KAFKA_AVAIL, PYTEST_KAFKA_ERROR) = _init_pytest_kafka()
if PYTEST_KAFKA_AVAIL:
    # Pull out the fixtures into this namespace
    from _utils.kafka import _kafka_consumer  # noqa: F401  pylint:disable=unused-import
    from _utils.kafka import kafka_server  # noqa: F401  pylint:disable=unused-import
    from _utils.kafka import zookeeper_proc  # noqa: F401  pylint:disable=unused-import


def pytest_addoption(parser: pytest.Parser):
    """
    Adds command line options for running specfic tests that are disabled by default
    """
    parser.addoption(
        "--run_slow",
        action="store_true",
        dest="run_slow",
        help="Run slow tests that would otherwise be skipped",
    )

    parser.addoption(
        "--run_kafka",
        action="store_true",
        dest="run_kafka",
        help="Run kafka tests that would otherwise be skipped",
    )

    parser.addoption(
        "--run_milvus",
        action="store_true",
        dest="run_milvus",
        help="Run milvus tests that would otherwise be skipped",
    )

    parser.addoption(
        "--run_benchmark",
        action="store_true",
        dest="run_benchmark",
        help="Run benchmark tests that would otherwise be skipped",
    )

    parser.addoption(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"],
        dest="log_level",
        help="A specific log level to use during testing. Defaults to WARNING if not set.",
    )

    parser.addoption(
        "--fail_missing",
        action="store_true",
        dest="fail_missing",
        help=("Tests requiring unmet dependencies are normally skipped. "
              "Setting this flag will instead cause them to be reported as a failure"),
    )


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """
    This function will add parameterizations for the `config` fixture depending on what types of config the test
    supports
    """

    # === use_cpp Parameterize ===
    use_cpp = metafunc.definition.get_closest_marker("use_cpp") is not None
    use_python = metafunc.definition.get_closest_marker("use_python") is not None

    use_cpp_param = pytest.param(True, marks=pytest.mark.use_cpp(added_by="generate_tests"), id="use_cpp")
    use_python_param = pytest.param(False, marks=pytest.mark.use_python(added_by="generate_tests"), id="use_python")

    _set_use_cpp_params = []

    if ("use_cpp" in metafunc.fixturenames):
        # Need to add some params since the fixture was requested

        # Add cpp unless use_cpp == True and use_python == False
        if not (use_python and not use_cpp):
            _set_use_cpp_params.append(use_cpp_param)

        # Add python unless use_cpp == False and use_python == True
        if not (not use_python and use_cpp):
            _set_use_cpp_params.append(use_python_param)

    elif (use_cpp and use_python):
        # Need to parameterize since we have multiple
        _set_use_cpp_params.extend([use_cpp_param, use_python_param])

    if (len(_set_use_cpp_params) > 0):
        metafunc.parametrize("_set_use_cpp", _set_use_cpp_params, indirect=True)

    # === df_type Parameterize ===
    if ("df_type" in metafunc.fixturenames):
        # df_type fixture was requested. Only parameterize if both marks or neither marks are found. Otherwise, the
        # fixture will determine it from the mark
        use_cudf = metafunc.definition.get_closest_marker("use_cudf") is not None
        use_pandas = metafunc.definition.get_closest_marker("use_pandas") is not None

        if (use_pandas == use_cudf):
            metafunc.parametrize(
                "df_type",
                [
                    pytest.param("cudf", marks=pytest.mark.use_cudf(added_by="generate_tests"), id="use_cudf"),
                    pytest.param("pandas", marks=pytest.mark.use_pandas(added_by="generate_tests"), id="use_pandas")
                ],
                indirect=True)


def pytest_runtest_setup(item):
    if (not item.config.getoption("--run_slow")):
        if (item.get_closest_marker("slow") is not None):
            pytest.skip("Skipping slow tests by default. Use --run_slow to enable")

    if (not item.config.getoption("--run_kafka")):
        if (item.get_closest_marker("kafka") is not None):
            pytest.skip("Skipping Kafka tests by default. Use --run_kafka to enable")

    if (not item.config.getoption("--run_milvus")):
        if (item.get_closest_marker("milvus") is not None):
            pytest.skip("Skipping milvus tests by default. Use --run_milvus to enable")

    if (not item.config.getoption("--run_benchmark")):
        if (item.get_closest_marker("benchmark") is not None):
            pytest.skip("Skipping benchmark tests by default. Use --run_benchmark to enable")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: typing.List[pytest.Item]):
    """
    To support old unittest style tests, try to determine the mark from the name
    """

    if config.getoption("--run_kafka") and not PYTEST_KAFKA_AVAIL:
        raise RuntimeError(f"--run_kafka requested but pytest_kafka not available due to: {PYTEST_KAFKA_ERROR}")

    for item in items:
        if "no_cpp" in item.nodeid and item.get_closest_marker("use_python") is None:
            item.add_marker(pytest.mark.use_python(added_in="collection_modifyitems"))
        elif "cpp" in item.nodeid and item.get_closest_marker("use_cpp") is None:
            item.add_marker(pytest.mark.use_cpp(added_in="collection_modifyitems"))

    def should_filter_test(item: pytest.Item):

        use_cpp = item.get_closest_marker("use_cpp")
        use_pandas = item.get_closest_marker("use_pandas")

        if (use_cpp and use_pandas):
            return False

        return True

    # Now delete tests with incompatible markers
    items[:] = [x for x in items if should_filter_test(x)]


def clear_handlers(logger):
    handlers = logger.handlers.copy()
    for handler in handlers:
        logger.removeHandler(handler)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    clear_handlers(logging.getLogger("morpheus"))
    clear_handlers(logging.getLogger())


# This fixture will be used by all tests.
@pytest.fixture(scope="function", autouse=True)
def _set_use_cpp(request: pytest.FixtureRequest):

    do_use_cpp: bool = True

    # Check for the param if this was indirectly set
    if (hasattr(request, "param") and isinstance(request.param, bool)):
        do_use_cpp = request.param
    else:
        # If not, check for the marker and use that
        use_cpp = request.node.get_closest_marker("use_cpp") is not None
        use_python = request.node.get_closest_marker("use_python") is not None

        if (use_cpp and use_python):
            raise RuntimeError(f"Both markers (use_cpp and use_python) were added to function {request.node.nodeid}. "
                               "Remove markers to support both.")

        # This will default to True or follow use_cpp
        do_use_cpp = not use_python

    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(do_use_cpp)

    yield do_use_cpp


# This fixture will be used by all tests.
@pytest.fixture(scope="function")
def use_cpp(_set_use_cpp: bool):

    # Just return the set value
    yield _set_use_cpp


@pytest.fixture(scope="function")
def config_only_cpp():
    """
    Use this fixture in unittest style tests to indicate a lack of support for C++. Use via
    `@pytest.mark.usefixtures("config_only_cpp")`
    """

    from morpheus.config import Config
    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(True)

    yield Config()


@pytest.fixture(scope="function")
def config_no_cpp():
    """
    Use this fixture in unittest style tests to indicate support for C++. Use via
    `@pytest.mark.usefixtures("config_no_cpp")`
    """

    from morpheus.config import Config
    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(False)

    yield Config()


@pytest.fixture(scope="function")
def df_type(request: pytest.FixtureRequest):

    df_type_str: typing.Literal["cudf", "pandas"]

    # Check for the param if this was indirectly set
    if (hasattr(request, "param")):
        assert request.param in ["pandas", "cudf"], "Invalid parameter for df_type"

        df_type_str = request.param
    else:
        # If not, check for the marker and use that
        use_pandas = request.node.get_closest_marker("use_pandas") is not None
        use_cudf = request.node.get_closest_marker("use_cudf") is not None

        if (use_pandas and use_cudf):
            raise RuntimeError(f"Both markers (use_cpp and use_python) were added to function {request.node.nodeid}. "
                               "Remove markers to support both.")

        # This will default to "cudf" or follow use_pandas
        df_type_str = "cudf" if not use_pandas else "pandas"

    yield df_type_str


@pytest.fixture(scope="function")
def config(use_cpp: bool):
    """
    For new pytest style tests, get the config by using this fixture. It will setup the config based on the marks set on
    the object. If no marks are added to the test, it will be parameterized for both C++ and python. For example,

    ```
    @pytest.mark.use_python
    def my_python_test(config: Config):
        ...
    ```
    """

    from morpheus.config import Config

    yield Config()


@pytest.fixture(scope="function")
def restore_environ():
    orig_vars = os.environ.copy()
    yield os.environ

    # Iterating over a copy of the keys as we will potentially be deleting keys in the loop
    for key in list(os.environ.keys()):
        orig_val = orig_vars.get(key)
        if orig_val is not None:
            os.environ[key] = orig_val
        else:
            del (os.environ[key])


@pytest.fixture(scope="function")
def restore_sys_path():
    orig_vars = sys.path.copy()
    yield sys.path
    sys.path = orig_vars


@pytest.fixture(scope="function")
def import_mod(request: pytest.FixtureRequest,
               restore_sys_path) -> typing.Generator[types.ModuleType | list[types.ModuleType], None, None]:
    # pylint: disable=missing-param-doc
    # pylint: disable=differing-param-doc
    # pylint: disable=missing-type-doc
    # pylint: disable=differing-type-doc
    """
    Allows direct import of a module by specifying its path. This is useful for testing examples that import modules in
    examples or other non-installed directories.

    Parameters
    ----------
    modules : str | list[str]
        The modules to import. Modules can be supplied as a list or multiple arguments.
    sys_path : str | int,
        When

    Yields
    ------
    Iterator[typing.Generator[types.ModuleType | list[types.ModuleType], None, None]]
        Imported modules. If more than one module is supplied, or the only argument is a list, the modules will be
        returned as a list.

    Example
    -------
    ```
    @pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'example/stage.py'))
    def test_python_test(import_mod: types.ModuleType):
        # Imported with sys.path.append(os.path.dirname(TEST_DIRS.examples_dir, 'example/stage.py'))
        ...

    @pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'example/stage.py'), sys_path=-2)
    def test_python_test(import_mod: types.ModuleType):
        # Imported with sys.path.append(os.path.join(TEST_DIRS.examples_dir, 'example/stage.py', '../..'))
        ...

    @pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'example/stage.py')], sys_path=TEST_DIRS.examples_dir)
    def test_python_test(import_mod: list[types.ModuleType]):
        # Imported with sys.path.append(TEST_DIRS.examples_dir)
        ...
    ```
    """

    marker = request.node.get_closest_marker("import_mod")
    if marker is not None:
        mod_paths = sum([x if isinstance(x, list) else [x] for x in marker.args], [])

        mod_kwargs = marker.kwargs

        is_list = len(marker.args) > 1 or isinstance(marker.args[0], list)

        modules = []
        module_names = []

        for mod_path in mod_paths:
            # Ensure everything is absolute to avoid issues with relative paths
            mod_path = os.path.abspath(mod_path)

            # See if its a file or directory
            is_file = os.path.isfile(mod_path)

            # Get the base directory that we should import from. If not specified, use the directory of the module
            sys_path = mod_kwargs.get("sys_path", os.path.dirname(mod_path))

            # If sys_path is an integer, use it to get the path relative to the module by number of directories. i.e. if
            # sys_path=-1, then sys_path=os.path.dirname(mod_path). If sys_path=-2, then
            # sys_path=os.path.dirname(os.path.dirname(mod_path))
            if (isinstance(sys_path, int)):
                sys_path = os.path.join("/", *mod_path.split(os.path.sep)[:sys_path])

            # Get the path relative to the sys_path, ignore the extension if its a file
            mod_name = os.path.relpath(mod_path if not is_file else os.path.splitext(mod_path)[0], start=sys_path)

            # Convert all / to .
            mod_name = mod_name.replace(os.path.sep, ".")

            # Add to the sys path so this can be imported
            sys.path.append(sys_path)

            try:

                # Import the module
                mod = importlib.import_module(mod_name)

                if (is_file):
                    assert mod.__file__ == mod_path

                modules.append(mod)
                module_names.append(mod_name)
            except ImportError as e:

                raise ImportError(f"Failed to import module {mod_path} as {mod_name} from path {sys_path}") from e

        # Only yield 1 if we only imported 1
        if (is_list):
            yield modules
        else:
            yield modules[0]

        # Un-import modules we previously imported, this allows for multiple examples to contain a `messages.py`
        for mod in module_names:
            sys.modules.pop(mod, None)

    else:
        raise ValueError("import_mod fixture requires setting paths in markers: "
                         "`@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing/messages.py')])`")


def _reload_modules(modules: typing.List[typing.Any]):
    for mod in modules:
        importlib.reload(mod)


@pytest.fixture(scope="function")
def reload_modules(request: pytest.FixtureRequest):
    marker = request.node.get_closest_marker("reload_modules")
    modules = []
    if marker is not None:
        modules = marker.args[0]
        if not isinstance(modules, list):
            modules = [modules]

    _reload_modules(modules)
    yield
    _reload_modules(modules)


@pytest.fixture(scope="function")
def manual_seed():
    """
    Seeds the random number generators for the stdlib, PyTorch and NumPy.
    By default this will seed with a value of `42`, however this fixture also yields the seed function allowing tests to
    call this a second time, or seed with a different value.

    Use this fixture to ensure repeatability of a test that depends on randomness.
    Note: PyTorch only garuntees determanism on a per-GPU basis, resulting in some tests that might not be portable
    across GPU models.
    """
    from morpheus.utils import seed as seed_utils

    def seed_fn(seed=42):
        seed_utils.manual_seed(seed)

    seed_fn()
    yield seed_fn


@pytest.fixture(scope="function")
def chdir_tmpdir(request: pytest.FixtureRequest, tmp_path):
    """
    Executes a test in the tmp_path directory
    """
    os.chdir(tmp_path)
    yield
    os.chdir(request.config.invocation_dir)


@pytest.fixture(scope="function")
def reset_plugin_manger():
    from morpheus.cli.plugin_manager import PluginManager
    PluginManager._singleton = None
    yield


@pytest.fixture(scope="function")
def reset_global_stage_registry():
    from morpheus.cli.stage_registry import GlobalStageRegistry
    from morpheus.cli.stage_registry import StageRegistry
    GlobalStageRegistry._global_registry = StageRegistry()
    yield


@pytest.fixture(scope="function")
def reset_plugins(reset_plugin_manger, reset_global_stage_registry):
    """
    Reset both the plugin manager and the global stage gregistry.
    Some of the tests for examples import modules dynamically, which in some cases can cause register_stage to be
    called more than once for the same stage.
    """
    yield


@pytest.fixture(scope="function")
def disable_gc():
    """
    Disable automatic garbage collection and enables debug stats for garbage collection for the duration of the test.
    This is useful for tests that require explicit control over when garbage collection occurs.
    """
    gc.set_debug(gc.DEBUG_STATS)
    gc.disable()
    yield
    gc.set_debug(0)
    gc.enable()


def wait_for_camouflage(host="localhost", port=8000, timeout=30):

    start_time = time.time()
    cur_time = start_time
    end_time = start_time + timeout

    url = f"http://{host}:{port}/ping"

    while cur_time - start_time <= timeout:
        timeout_epoch = min(cur_time + 2.0, end_time)

        try:
            request_timeout = max(timeout_epoch - cur_time, 0.1)
            resp = requests.get(url, timeout=request_timeout)

            if (resp.status_code == 200):
                if (resp.json()['message'] == 'I am alive.'):
                    return True

                warnings.warn(("Camoflage returned status 200 but had incorrect response JSON. Continuing to wait. "
                               "Response JSON:\n%s"),
                              resp.json())

        except Exception:
            pass

        # Sleep up to the end time or max 1 second
        time.sleep(max(timeout_epoch - time.time(), 0.0))

        # Update current time
        cur_time = time.time()

    return False


def _set_pdeathsig(sig=signal.SIGTERM):
    """
    Helper function to ensure once parent process exits, its child processes will automatically die
    """

    def prctl_fn():
        libc = ctypes.CDLL("libc.so.6")
        return libc.prctl(1, sig)

    return prctl_fn


def _start_camouflage(
        root_dir: str,
        host: str = "localhost",
        port: int = 8000) -> typing.Tuple[bool, typing.Optional[subprocess.Popen], typing.Optional[typing.IO]]:
    logger = logging.getLogger(f"morpheus.{__name__}")
    startup_timeout = 30

    launch_camouflage = os.environ.get('MORPHEUS_NO_LAUNCH_CAMOUFLAGE') is None
    is_running = False

    # First, check to see if camoflage is already open
    if (launch_camouflage):
        is_running = wait_for_camouflage(host=host, port=port, timeout=0.0)

        if (is_running):
            logger.warning("Camoflage already running. Skipping startup")
            launch_camouflage = False
            is_running = True

    # Actually launch camoflague
    if launch_camouflage:
        popen = None
        console_log_fh = None
        try:
            # pylint: disable=subprocess-popen-preexec-fn,consider-using-with
            # We currently don't have control over camouflage's console logger
            # https://github.com/testinggospels/camouflage/issues/244
            console_log = os.path.join(root_dir, 'console.log')
            camouflage_log = os.path.join(root_dir, 'camouflage.log')
            console_log_fh = open(console_log, 'w', encoding='utf-8')
            popen = subprocess.Popen(["camouflage", "--config", "config.yml"],
                                     cwd=root_dir,
                                     stderr=subprocess.STDOUT,
                                     stdout=console_log_fh,
                                     preexec_fn=_set_pdeathsig(signal.SIGTERM))
            # pylint: enable=subprocess-popen-preexec-fn,consider-using-with

            logger.info("Launched camouflage in %s with pid: %s", root_dir, popen.pid)

            def read_logs():
                for log_file in (console_log, camouflage_log):
                    if os.path.exists(log_file):
                        with open(log_file, 'r', encoding='utf-8') as f:
                            logger.error("%s:\n%s", log_file, f.read())
                            # We only need to display the first log file that exists
                            return

            if not wait_for_camouflage(host=host, port=port, timeout=startup_timeout):
                if console_log_fh is not None:
                    console_log_fh.close()

                read_logs()

                if popen.poll() is not None:
                    raise RuntimeError(f"camouflage server exited with status code={popen.poll()}")

                raise RuntimeError("Failed to launch camouflage server")

            # Must have been started by this point
            return (True, popen, console_log_fh)

        except Exception:
            # Log the error and rethrow
            logger.exception("Error launching camouflage")
            if popen is not None:
                _stop_camouflage(popen, console_log_fh=console_log_fh)
            raise

    else:

        return (is_running, None, None)


def _stop_camouflage(popen: subprocess.Popen, shutdown_timeout: int = 5, console_log_fh: typing.IO = None):
    logger = logging.getLogger(f"morpheus.{__name__}")

    logger.info("Killing camouflage with pid %s", popen.pid)

    elapsed_time = 0.0
    sleep_time = 0.1
    stopped = False

    # It takes a little while to shutdown
    while not stopped and elapsed_time < shutdown_timeout:
        popen.kill()
        stopped = (popen.poll() is not None)
        if not stopped:
            time.sleep(sleep_time)
            elapsed_time += sleep_time

    if console_log_fh is not None:
        console_log_fh.close()


@pytest.fixture(scope="session")
def _triton_camouflage_is_running():
    """
    Responsible for actually starting and shutting down Camouflage running with the mocks in the `mock_triton_server`
    dir. This has the scope of 'session' so we only start/stop Camouflage once per testing session. This fixture should
    not be used directly. Instead use `launch_mock_triton`

    Yields
    ------
    bool
        Whether or not we are using Camouflage or an actual Triton server
    """

    from _utils import TEST_DIRS

    root_dir = TEST_DIRS.mock_triton_servers_dir
    (is_running, popen, console_log_fh) = _start_camouflage(root_dir=root_dir, port=8000)
    yield is_running
    if popen is not None:
        _stop_camouflage(popen, console_log_fh=console_log_fh)


@pytest.fixture(scope="session")
def _rest_camouflage_is_running():
    """
    Responsible for actually starting and shutting down Camouflage running with the mocks in the `mock_rest_server` dir.
    This has the scope of 'session' so we only start/stop Camouflage once per testing session. This fixture should not
    be used directly. Instead use `launch_mock_rest`

    Yields
    ------
    bool
        Whether or not we are using Camouflage or an actual Rest server
    """

    from _utils import TEST_DIRS

    root_dir = TEST_DIRS.mock_rest_server
    (is_running, popen, console_log_fh) = _start_camouflage(root_dir=root_dir, port=8080)

    yield is_running
    if popen is not None:
        _stop_camouflage(popen, console_log_fh=console_log_fh)


@pytest.fixture(scope="function")
def launch_mock_triton(_triton_camouflage_is_running):
    """
    Launches a mock triton server using camouflage (https://testinggospels.github.io/camouflage/).

    This function will wait for up to `timeout` seconds for camoflauge to startup

    This function is a no-op if the `MORPHEUS_NO_LAUNCH_CAMOUFLAGE` environment variable is defined, which can
    be useful during test development to run camouflage by hand.
    """

    # Check if we are using Camouflage or not. If so, send the reset command to reset the state
    if _triton_camouflage_is_running:
        # Reset the mock server (necessary to set counters = 0)
        resp = requests.post("http://localhost:8000/reset", timeout=2.0)

        assert resp.ok, "Failed to reset Camouflage server state"

    yield


@pytest.fixture(scope="function")
def mock_rest_server(_rest_camouflage_is_running):
    """
    Launches a mock rest server using camouflage (https://testinggospels.github.io/camouflage/).

    This function will wait for up to `timeout` seconds for camoflauge to startup

    This function is a no-op if the `MORPHEUS_NO_LAUNCH_CAMOUFLAGE` environment variable is defined, which can
    be useful during test development to run camouflage by hand.

    yields url to the mock rest server
    """

    # Check if we are using Camouflage or not.
    assert _rest_camouflage_is_running

    yield "http://localhost:8080"


@pytest.fixture(scope="session", autouse=True)
def configure_tests_logging(pytestconfig: pytest.Config):
    """
    Sets the base logging settings for the entire test suite to ensure logs are generated. Automatically detects if a
    debugger is attached and lowers the logging level to DEBUG.
    """
    from morpheus.utils.logger import configure_logging

    log_level = logging.WARNING

    # Check if a debugger is attached. If so, choose DEBUG for the logging level. Otherwise, only WARN
    trace_func = sys.gettrace()

    if (trace_func is not None):
        trace_module = getattr(trace_func, "__module__", None)

        if (trace_module is not None and trace_module.find("pydevd") != -1):
            log_level = logging.DEBUG

    config_log_level = pytestconfig.getoption("log_level")

    # Overwrite the logging level if specified
    if (config_log_level is not None):
        log_level = logging.getLevelName(config_log_level)

    configure_logging(log_level=log_level)


def _wrap_set_log_level(log_level: int):
    from morpheus.utils.logger import set_log_level

    # Save the previous logging level
    old_level = set_log_level(log_level)

    yield

    set_log_level(old_level)


@pytest.fixture(scope="session")
def fail_missing(pytestconfig: pytest.Config) -> bool:
    """
    Returns the value of the `fail_missing` flag, when false tests requiring unmet dependencies will be skipped, when
    True they will fail.
    """
    yield pytestconfig.getoption("fail_missing")


# ==== Logging Fixtures ====
@pytest.fixture(scope="function")
def reset_loglevel():
    """
    Fixture restores the log level after running the given test.
    """
    import mrc

    from morpheus.utils.logger import set_log_level

    old_level = mrc.logging.get_level()

    yield

    set_log_level(old_level)


@pytest.fixture(scope="function")
def loglevel_debug():
    """
    Sets the logging level to `logging.DEBUG` for this function only.
    """
    _wrap_set_log_level(logging.DEBUG)


@pytest.fixture(scope="function")
def loglevel_info():
    """
    Sets the logging level to `logging.INFO` for this function only.
    """
    _wrap_set_log_level(logging.INFO)


@pytest.fixture(scope="function")
def loglevel_warn():
    """
    Sets the logging level to `logging.WARN` for this function only.
    """
    _wrap_set_log_level(logging.WARN)


@pytest.fixture(scope="function")
def loglevel_error():
    """
    Sets the logging level to `logging.ERROR` for this function only.
    """
    _wrap_set_log_level(logging.ERROR)


@pytest.fixture(scope="function")
def loglevel_fatal():
    """
    Sets the logging level to `logging.FATAL` for this function only.
    """
    _wrap_set_log_level(logging.FATAL)


# ==== DataFrame Fixtures ====
@pytest.fixture(scope="function")
def dataset(df_type: typing.Literal['cudf', 'pandas']):
    """
    Yields a DatasetLoader instance with `df_type` as the default DataFrame type.
    Users of this fixture can still explicitly request either a cudf or pandas dataframe with the `cudf` and `pandas`
    properties:
    ```
    def test_something(dataset: DatasetManager):
        df = dataset["filter_probs.csv"]  # type will match the df_type parameter
        if dataset.default_df_type == 'pandas':
            assert isinstance(df, pd.DataFrame)
        else:
            assert isinstance(df, cudf.DataFrame)

        pdf = dataset.pandas["filter_probs.csv"]
        cdf = dataset.cudf["filter_probs.csv"]

    ```

    A test that requests this fixture will parameterize on the type of DataFrame returned by the DatasetManager.
    If a test requests both this fixture and the `use_cpp` fixture, or indirectly via the `config` fixture, then
    the test will parameterize over both df_type:[cudf, pandas] and use_cpp[True, False]. However it will remove the
    df_type=pandas & use_cpp=True combinations as this will cause an unsupported usage of Pandas dataframes with the
    C++ implementation of message classes.

    This behavior can also be overridden by using the `use_cudf`, `use_pandas`, `use_cpp` or `use_pandas` marks ex:
    ```
    # This test will only run once with C++ enabled and cudf dataframes
    @pytest.mark.use_cpp
    def test something(dataset: DatasetManager):
    ...
    # This test will run once for each dataframe type, with C++ disabled both times
    @pytest.mark.use_python
    import sysdf dataframes both times
    @pytest.mark.use_cudf
    def test something(use_cpp: bool, dataset: DatasetManager):
    ...
    # This test will run only once
    @pytest.mark.use_cudf
    @pytest.mark.use_python
    def test something(dataset: DatasetManager):
    ...
    # This test creates an incompatible combination and will raise a RuntimeError without being executed
    @pytest.mark.use_pandas
    @pytest.mark.use_cpp
    def test something(dataset: DatasetManager):
    ```

    Users who don't want to parametarize over the DataFrame should use the `dataset_pandas` or `dataset_cudf` fixtures.
    """
    from _utils import dataset_manager
    yield dataset_manager.DatasetManager(df_type=df_type)


@pytest.fixture(scope="function")
def dataset_pandas():
    """
    Yields a DatasetLoader instance with pandas as the default DataFrame type.

    Note: This fixture won't prevent a user from writing a test requiring C++ mode execution and requesting Pandas
    dataframes. This is quite useful for tests like `tests/test_add_scores_stage_pipe.py` where we want to test with
    both Python & C++ executions, but we use Pandas to build up the expected DataFrame to validate the test against.

    In addition to this, users can use this fixture to explicitly request a cudf Dataframe as well, allowing for a test
    that looks like:
    ```
    @pytest.mark.use_cpp
    def test_something(dataset_pandas: DatasetManager):
        input_df = dataset_pandas.cudf["filter_probs.csv"] # Feed our source stage a cudf DF

        # Perform pandas transformations to mimic the add scores stage
        expected_df = dataset["filter_probs.csv"]
        expected_df = expected_df.rename(columns=dict(zip(expected_df.columns, class_labels)))
    ```
    """
    from _utils import dataset_manager
    yield dataset_manager.DatasetManager(df_type='pandas')


@pytest.fixture(scope="function")
def dataset_cudf():
    """
    Yields a DatasetLoader instance with cudf as the default DataFrame type.

    Users who wish to have both cudf and pandas DataFrames can do so with this fixture and using the `pandas` property:
    def test_something(dataset_cudf: DatasetManager):
        cdf = dataset_cudf["filter_probs.csv"]
        pdf = dataset_cudf.pandas["filter_probs.csv"]
    """
    from _utils import dataset_manager
    yield dataset_manager.DatasetManager(df_type='cudf')


@pytest.fixture(scope="function")
def filter_probs_df(dataset, use_cpp: bool):
    """
    Shortcut fixture for loading the filter_probs.csv dataset.

    Unless your test uses the `use_pandas` or `use_cudf` marks this fixture will parametarize over the two dataframe
    types. Similarly unless your test uses the `use_cpp` or `use_python` marks this fixture will also parametarize over
    that as well, while excluding the combination of C++ execution and Pandas dataframes.
    """
    yield dataset["filter_probs.csv"]


def _get_random_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sckt:
        sckt.bind(('', 0))
        return sckt.getsockname()[1]


@pytest.fixture(scope="session")
def milvus_server_uri(tmp_path_factory):
    """
    Pytest fixture to start and stop a Milvus server and provide its URI for testing.
    Due to the high startup time for Milvus users can optionally start a Milvus server before running tests and
    define a `MORPHEUS_MILVUS_URI` environment variable to use that server instead of starting a new one.

    This fixture starts a Milvus server, retrieves its URI (Uniform Resource Identifier), and provides
    the URI as a yield value to the tests using this fixture. After all tests in the module are
    completed, the Milvus server is stopped.
    """
    logger = logging.getLogger(f"morpheus.{__name__}")

    uri = os.environ.get('MORPHEUS_MILVUS_URI')
    if uri is not None:
        yield uri

    else:
        from milvus import default_server

        # Milvus checks for already bound ports but it doesnt seem to work for webservice_port. Use a random one
        default_server.webservice_port = _get_random_port()
        with default_server:
            default_server.set_base_dir(tmp_path_factory.mktemp("milvus_store"))

            host = default_server.server_address
            port = default_server.listen_port
            uri = f"http://{host}:{port}"

            logger.info("Started Milvus at: %s", uri)

            yield uri


@pytest.fixture(scope="session", name="milvus_data")
def milvus_data_fixture():
    inital_data = [{"id": i, "embedding": [i / 10.0] * 3, "age": 25 + i} for i in range(10)]
    yield inital_data


@pytest.fixture(scope="session", name="idx_part_collection_config")
def idx_part_collection_config_fixture():
    from _utils import load_json_file
    yield load_json_file(filename="service/milvus_idx_part_collection_conf.json")


@pytest.fixture(scope="session", name="simple_collection_config")
def simple_collection_config_fixture():
    from _utils import load_json_file
    yield load_json_file(filename="service/milvus_simple_collection_conf.json")


@pytest.fixture(name="nemollm", scope='session')
def nemollm_fixture(fail_missing: bool):
    """
    Fixture to ensure nemollm is installed
    """
    skip_reason = ("Tests for the NeMoLLMService require the nemollm package to be installed, to install this run:\n"
                   "`mamba install -n base -c conda-forge conda-merge`\n"
                   "`conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml "
                   "  docker/conda/environments/cuda${CUDA_VER}_examples.yml"
                   "  > .tmp/merged.yml && mamba env update -n morpheus --file .tmp/merged.yml`")
    yield import_or_skip("nemollm", reason=skip_reason, fail_missing=fail_missing)


@pytest.fixture(name="openai", scope='session')
def openai_fixture(fail_missing: bool):
    """
    Fixture to ensure openai is installed
    """
    skip_reason = ("Tests for the OpenAIChatService require the openai package to be installed, to install this run:\n"
                   "`mamba install -n base -c conda-forge conda-merge`\n"
                   "`conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml "
                   "  docker/conda/environments/cuda${CUDA_VER}_examples.yml"
                   "  > .tmp/merged.yml && mamba env update -n morpheus --file .tmp/merged.yml`")
    yield import_or_skip("openai", reason=skip_reason, fail_missing=fail_missing)


@pytest.mark.usefixtures("openai")
@pytest.fixture(name="chat_completion")
def chat_completion_fixture():
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion import ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    chat_completion = ChatCompletion(id='chatcmpl-8Xurraq4bdGlFDRvJMx9C2TsVkyQM',
                                     choices=[
                                         Choice(finish_reason='stop',
                                                index=0,
                                                logprobs=None,
                                                message=ChatCompletionMessage(content='test_output',
                                                                              role='assistant',
                                                                              function_call=None,
                                                                              tool_calls=None))
                                     ],
                                     created=1703094115,
                                     model='gpt-3.5-turbo',
                                     object='chat.completion')
    yield chat_completion


@pytest.mark.usefixtures("openai")
@pytest.fixture(name="mock_openai")
def mock_openai_fixture(chat_completion):
    with mock.patch("openai.OpenAI") as mock_openai:
        mock_openai_instance = mock.MagicMock()
        mock_create = mock.MagicMock()
        mock_create.return_value = chat_completion.copy()
        mock_openai_instance.chat.completions.create = mock_create

        mock_openai.return_value = mock_openai_instance

        yield mock_openai_instance


@pytest.mark.usefixtures("openai")
@pytest.fixture(name="mock_async_openai")
def mock_async_openai_fixture(chat_completion):
    with mock.patch("openai.AsyncOpenAI") as mock_async_openai:
        mock_async_openai_instance = mock.MagicMock()
        mock_create = mock.AsyncMock()
        mock_create.return_value = chat_completion.copy()
        mock_async_openai_instance.chat.completions.create = mock_create

        mock_async_openai.return_value = mock_async_openai_instance

        yield mock_async_openai_instance


@pytest.mark.usefixtures("nemollm")
@pytest.fixture(name="mock_nemollm")
def mock_nemollm_fixture():
    with mock.patch("nemollm.NemoLLM") as mock_nemollm:
        mock_nemollm.return_value = mock_nemollm
        mock_nemollm.generate_multiple.return_value = ["test_output"]
        mock_nemollm.post_process_generate_response.return_value = {"text": "test_output"}

        yield mock_nemollm
