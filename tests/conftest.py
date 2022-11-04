# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import importlib
import logging
import os
import signal
import subprocess
import time
import typing
import warnings
from collections import namedtuple
from functools import partial

import pytest
import requests

# actual topic names not important, but we will need two of them.
KAFKA_TOPICS = namedtuple('KAFKA_TOPICS', ['input_topic', 'output_topic'])('morpheus_input_topic',
                                                                           'morpheus_output_topic')

zookeeper_proc = None
kafka_server = None
kafka_consumer = None
pytest_kafka_setup_error = None


def init_pytest_kafka():
    """
    pytest_kafka is currently required to be installed manually, along with a download of Kafka and a functional JDK.
    Since the Kafka tests don't run by default, we will silently fail to initialize unless --run_kafka is enabled.

    Issue #9 should make the instalation of Kafka simpler:
    https://gitlab.com/karolinepauls/pytest-kafka/-/issues/9
    """
    global zookeeper_proc, kafka_server, kafka_consumer, pytest_kafka_setup_error
    try:
        import pytest_kafka
        os.environ['KAFKA_OPTS'] = "-Djava.net.preferIPv4Stack=True"
        # Initialize pytest_kafka fixtures following the recomendations in:
        # https://gitlab.com/karolinepauls/pytest-kafka/-/blob/master/README.rst
        KAFKA_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(pytest_kafka.__file__)), 'kafka/bin/')
        KAFKA_BIN = os.path.join(KAFKA_SCRIPTS, 'kafka-server-start.sh')
        ZOOKEEPER_BIN = os.path.join(KAFKA_SCRIPTS, 'zookeeper-server-start.sh')

        for kafka_script in (KAFKA_BIN, ZOOKEEPER_BIN):
            if not os.path.exists(kafka_script):
                raise RuntimeError("Required Kafka script not found: {}".format(kafka_script))

        teardown_fn = partial(pytest_kafka.terminate, signal_fn=subprocess.Popen.kill)
        zookeeper_proc = pytest_kafka.make_zookeeper_process(ZOOKEEPER_BIN, teardown_fn=teardown_fn)
        kafka_server = pytest_kafka.make_kafka_server(KAFKA_BIN, 'zookeeper_proc', teardown_fn=teardown_fn)
        kafka_consumer = pytest_kafka.make_kafka_consumer('kafka_server',
                                                          group_id='morpheus_unittest_reader',
                                                          client_id='morpheus_unittest_reader',
                                                          seek_to_beginning=True,
                                                          kafka_topics=[KAFKA_TOPICS.output_topic])

        return True
    except Exception as e:
        pytest_kafka_setup_error = e
        return False


pytest_kafka_avail = init_pytest_kafka()


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


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """
    This function will add parameterizations for the `config` fixture depending on what types of config the test
    supports
    """

    # Only care about the config fixture
    if ("config" not in metafunc.fixturenames):
        return

    use_cpp = metafunc.definition.get_closest_marker("use_cpp") is not None
    use_python = metafunc.definition.get_closest_marker("use_python") is not None

    if (use_cpp and use_python):
        raise RuntimeError(
            "Both markers (use_cpp and use_python) were added to function {}. Remove markers to support both.".format(
                metafunc.definition.nodeid))
    elif (not use_cpp and not use_python):
        # Add the markers to the parameters
        metafunc.parametrize("config",
                             [
                                 pytest.param(True, marks=pytest.mark.use_cpp, id="use_cpp"),
                                 pytest.param(False, marks=pytest.mark.use_python, id="use_python")
                             ],
                             indirect=True)


def pytest_runtest_setup(item):
    if (not item.config.getoption("--run_slow")):
        if (item.get_closest_marker("slow") is not None):
            pytest.skip("Skipping slow tests by default. Use --run_slow to enable")

    if (not item.config.getoption("--run_kafka")):
        if (item.get_closest_marker("kafka") is not None):
            pytest.skip("Skipping Kafka tests by default. Use --run_kafka to enable")


def pytest_collection_modifyitems(config, items):
    """
    To support old unittest style tests, try to determine the mark from the name
    """

    if config.getoption("--run_kafka") and not pytest_kafka_avail:
        raise RuntimeError(
            "--run_kafka requested but pytest_kafka not available due to: {}".format(pytest_kafka_setup_error))

    for item in items:
        if "no_cpp" in item.nodeid:
            item.add_marker(pytest.mark.use_python)
        elif "cpp" in item.nodeid:
            item.add_marker(pytest.mark.use_cpp)


def clear_handlers(logger):
    handlers = logger.handlers.copy()
    for h in handlers:
        logger.removeHandler(h)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    clear_handlers(logging.getLogger("morpheus"))
    clear_handlers(logging.getLogger())


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
def config(request: pytest.FixtureRequest):
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
    from morpheus.config import CppConfig

    if (not hasattr(request, "param")):
        use_cpp = request.node.get_closest_marker("use_cpp") is not None
        use_python = request.node.get_closest_marker("use_python") is not None

        assert use_cpp != use_python, "Invalid config"

        CppConfig.set_should_use_cpp(True if use_cpp else False)

    else:
        CppConfig.set_should_use_cpp(True if request.param else False)

    yield Config()


@pytest.fixture(scope="function")
def kafka_topics():
    """
    Returns a named tuple of Kafka topic names in the form of (input_topic, output_topic)
    """
    yield KAFKA_TOPICS


@pytest.fixture(scope="function")
def kafka_bootstrap_servers(kafka_server: typing.Tuple[subprocess.Popen, int]):
    """
    Used by tests that require both an input and an output topic
    """
    kafka_port = kafka_server[1]
    yield "localhost:{}".format(kafka_port)


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
def reload_modules(request: pytest.FixtureRequest):
    marker = request.node.get_closest_marker("reload_modules")
    yield

    if marker is not None:
        modules = marker.args[0]
        if not isinstance(modules, list):
            modules = [modules]

        for mod in modules:
            importlib.reload(mod)


@pytest.fixture(scope="function")
def chdir_tmpdir(request: pytest.FixtureRequest, tmp_path):
    """
    Executes a test in the tmp_path directory
    """
    os.chdir(tmp_path)
    yield
    os.chdir(request.config.invocation_dir)


def wait_for_camouflage(host="localhost", port=8000, timeout=5):

    start_time = time.time()
    cur_time = start_time
    end_time = start_time + timeout

    url = "http://{}:{}/ping".format(host, port)

    while cur_time - start_time <= timeout:
        timeout_epoch = min(cur_time + 1.0, end_time)

        try:
            request_timeout = max(timeout_epoch - cur_time, 0.1)
            r = requests.get(url, timeout=request_timeout)

            if (r.status_code == 200):
                if (r.json()['message'] == 'I am alive.'):
                    return True
                else:
                    warnings.warn(
                        "Camoflage returned status 200 but had incorrect response JSON. Continuing to wait. Response JSON:\n{}"
                        .format(r.json()))

        except Exception:
            pass

        # Sleep up to the end time or max 1 second
        time.sleep(max(timeout_epoch - time.time(), 0.0))

        # Update current time
        cur_time = time.time()

    return False


def _set_pdeathsig(sig=signal.SIGTERM):
    """
    Helper function to ensure once parent process exits, its childrent processes will automatically die
    """

    def prctl_fn():
        libc = ctypes.CDLL("libc.so.6")
        return libc.prctl(1, sig)

    return prctl_fn


@pytest.fixture(scope="session")
def _camouflage_is_running():
    """
    Responsible for actually starting and shutting down Camouflage. This has the scope of 'session' so we only
    start/stop Camouflage once per testing session. Should not be used directly. Instead use `launch_mock_triton`

    Yields
    ------
    bool
        Whether or not we are using Camouflage or an actual Triton server
    """

    from utils import TEST_DIRS

    root_dir = TEST_DIRS.mock_triton_servers_dir
    startup_timeout = 5
    shutdown_timeout = 5

    launch_camouflage = os.environ.get('MORPHEUS_NO_LAUNCH_CAMOUFLAGE') is None
    is_running = False

    # First, check to see if camoflage is already open
    if (launch_camouflage):
        is_running = wait_for_camouflage(timeout=0.0)

        if (is_running):
            logging.warning("Camoflage already running. Skipping startup")
            launch_camouflage = False
            is_running = True

    # Actually launch camoflague
    if launch_camouflage:
        popen = subprocess.Popen(["camouflage", "--config", "config.yml"],
                                 cwd=root_dir,
                                 stderr=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL,
                                 preexec_fn=_set_pdeathsig(signal.SIGTERM))

        logging.info("Launched camouflage in %s with pid: %s", root_dir, popen.pid)

        if startup_timeout > 0:
            if not wait_for_camouflage(timeout=startup_timeout):

                if popen.poll() is not None:
                    raise RuntimeError("camouflage server exited with status code={} details in: {}".format(
                        popen.poll(), os.path.join(root_dir, 'camouflage.log')))

                raise RuntimeError("Failed to launch camouflage server")

        yield is_running

        logging.info("Killing pid {}".format(popen.pid))

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

    else:

        yield is_running


@pytest.fixture(scope="function")
def launch_mock_triton(_camouflage_is_running):
    """
    Launches a mock triton server using camouflage (https://testinggospels.github.io/camouflage/) with a package
    rooted at `root_dir` and configured with `config`.

    This function will wait for up to `timeout` seconds for camoflauge to startup

    This function is a no-op if the `MORPHEUS_NO_LAUNCH_CAMOUFLAGE` environment variable is defined, which can
    be useful during test development to run camouflage by hand.
    """

    # Check if we are using Camouflage or not. If so, send the reset command to reset the state
    if _camouflage_is_running:
        # Reset the mock server (necessary to set counters = 0)
        r = requests.post("http://localhost:8000/reset", timeout=2.0)

        assert r.ok, "Failed to reset Camouflage server state"

    yield
