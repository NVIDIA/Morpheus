# Copyright (c) 2023-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf

from morpheus.parsers.splunk_notable_parser import SplunkNotableParser

TEST_DATA1 = '1566345812.924, search_name="Test Search Name", orig_time="1566345812.924", ' \
             'info_max_time="1566346500.000000000", info_min_time="1566345300.000000000", ' \
             'info_search_time="1566305689.361160000", message.description="Test Message Description", ' \
             'message.hostname="msg.test.hostname", message.ip="100.100.100.123", ' \
             'message.user_name="user@test.com", severity="info", urgency="medium"'
TEST_DATA2 = '1548772230, search_name="Test Search Name 2", signature="Android.Adware.Batmobil", ' \
             'signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", ' \
             'src="10.01.01.1", src_ip="10.01.01.123", src_ip="10.01.01.1, count="19", ' \
             'info_max_time="1548772200.000000000", info_min_time="1548599400.000000000", ' \
             'info_search_time="1548772206.179561000", info_sid="test-info-sid", lastTime="1548771235", ' \
             'orig_raw="<164>fenotify-113908.warning: CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|' \
             'rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 UTC src=10.01.01.123 dest="10.01.01.122" '\
             'request=http://test.com/test.php cs1Label=sname cs1=Android.PUP.Downloader act=notified ' \
             'dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f dmac=1a:2b:3c:4d:5e:7g spt=49458 ' \
             'dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc msg=risk ware detected:57007 ' \
             'proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test cs6Label=channel cs6=POST ' \
             '/multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: Dalvik/2.1.0 (Linux; U; ' \
             'Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: ' \
             'Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,' \
             '"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"'
TEST_DATA3 = '1548234811, search_name="Test Search Name 3", signature="Android.Adware.Batmobil", ' \
             'signature="Android.Adware.Dlv", signature="Android.PUP.Downloader", src="10.01.01.123", ' \
             'src="10.01.01.1", count="19", info_max_time="1548234811.000000000", ' \
             'info_min_time="1548599400.000000000", info_search_time="1548772206.179561000", ' \
             'info_sid="test-info-sid", lastTime="1548771235", orig_raw="<164>fenotify-113908.warning: ' \
             'CEF:0|FireEye|MPS|1.2.3.123|RC|riskware-callback|1|rt=Jan 29 2019 14:13:55 UTC end=Jan 29 2019 14:13:55 '\
             'UTC src=10.01.01.123 dest="10.01.01.122" request=http://test.com/test.php cs1Label=sname ' \
             'cs1=Android.PUP.Downloader act=notified dvc=10.01.01.2 dvchost=fireeye.ban2-in smac=1a:2b:3c:4d:5e:6f ' \
             'dmac=1a:2b:3c:4d:5e:7g spt=49458 dpt=80 cn1Label=vlan cn1=0 externalId=123456 devicePayloadId=123abc ' \
             'msg=risk ware detected:57007 proto=tcp cs4Label=link cs4=https://fireeye.test/notification_url/test ' \
             'cs6Label=channel cs6=POST /multiadctrol.php HTTP/1.1::~~Content-type: application/json::~~User-Agent: ' \
             'Dalvik/2.1.0 (Linux; U; Android 8.0.0; SM-G611F Build/R16NW)::~~Host: test.hostname::~~Connection: ' \
             'Keep-Alive::~~Accept-Encoding: gzip::~~Content-Length: 85::~~::~~[{"android_id":"123abc","isnew":0,' \
             '"m_ch":"123","s_ch":"1","ver_c":"342\\"}] \n\\\\x00", orig_sourcetype="source", src_subnet="12.34.56"'
TEST_DATA4 = '1566345700, search_name="Endpoint - Brute Force against Known User - Rule", orig_source="100.20.2.21", ' \
             'orig_source="FEDEX-MA", orig_source="localhost.com", failure="1104", first="Pattrick", ' \
             'identity="pjame", info_max_time="1546382400.000000000", info_min_time="1546378800.000000000", ' \
             'info_search_time="1546382850.589570000", success="8", user="pjame'
TEST_DATA5 = '1566345700, search_name="Manual Notable Event - Rule", _time="1554290847", ' \
             'app="SplunkEnterpriseSecuritySuite", creator="test@nvidia.com", info_max_time="+Infinity", ' \
             'info_min_time="0.000", info_search_time="1554290847.423961000", owner="test@nvidia.com", ' \
             'rule_description="FireEye NX alert for Incident Review with Major Severity", ' \
             'rule_title="FireEye NX alert for Incident Review(Majr)", security_domain="endpoint", status="0", ' \
             'urgency="medium"'
TEST_DATA6 = '1566345700, search_name="Endpoint - FireEye NX alert for Incident Review (Minor) - Rule", ' \
             'category="riskware-callback", dest_ip="10.15.90.150", occurred="Mar 09 2019 02:36:00 UTC", ' \
             'signature="Android.Adware.Batmobil", src_ip="10.15.90.151", dest_port="80", src_port="40472", ' \
             'orig_time="1552098960", info_max_time="1552099380.000000000", info_min_time="1552098780.000000000", ' \
             'info_search_time="1552052094.393543000", severity="minr", src_host="ip-10.5.13.compute.internal"'
TEST_DATA7 = '1566345700, search_name=\\"Endpoint - Host With Malware Detected (Quarantined or Waived) - ' \
             'Rule\\", count=\\"1\\", dest=\\"TEST-01\\", dest_priority=\\"medium\\", ' \
             'info_max_time=\\"1511389440.000000000\\", info_min_time=\\"1511388840.000000000\\", ' \
             'info_search_time=\\"1511389197.841039000\\", info_sid=\\"rt_scheduler_dGNhcnJvbGxAbnZpZGlhLmNvbQ__' \
             'SplunkEnterpriseSecuritySuite__RMD5c5145919d43bdffc_at_1511389196_22323\\", ' \
             'lastTime=\\"1511388996.202094\\"'


def test_splunk_notable_parser():
    """Test splunk notable parsing"""

    snp = SplunkNotableParser()

    test_input_1 = cudf.Series([TEST_DATA1])
    test_output_df_1 = snp.parse(test_input_1)
    assert len(test_output_df_1.columns) == 23
    assert test_output_df_1["time"][0] == "1566345812.924"
    assert test_output_df_1["search_name"][0] == "Test Search Name"
    assert test_output_df_1["orig_time"][0] == "1566345812.924"
    assert test_output_df_1["urgency"][0] == "medium"
    assert test_output_df_1["user"][0] == ""
    assert test_output_df_1["owner"][0] == ""
    assert test_output_df_1["security_domain"][0] == ""
    assert test_output_df_1["severity"][0] == "info"
    assert test_output_df_1["src_ip"][0] == ""
    assert test_output_df_1["src_mac"][0] == ""
    assert test_output_df_1["src_port"][0] == ""
    assert test_output_df_1["dest_ip"][0] == ""
    assert test_output_df_1["dest_port"][0] == ""
    assert test_output_df_1["dest_mac"][0] == ""
    assert test_output_df_1["dest_priority"][0] == ""
    assert test_output_df_1["device_name"][0] == ""
    assert test_output_df_1["event_name"][0] == ""
    assert test_output_df_1["event_type"][0] == ""
    assert test_output_df_1["ip_address"][0] == ""
    assert test_output_df_1["message_ip"][0] == "100.100.100.123"
    assert test_output_df_1["message_username"][0] == "user@test.com"
    assert test_output_df_1["message_hostname"][0] == "msg.test.hostname"
    assert test_output_df_1["message_description"][0] == "Test Message Description"

    test_input_2 = cudf.Series([TEST_DATA2])
    test_output_df_2 = snp.parse(test_input_2)
    assert len(test_output_df_2.columns) == 23
    assert test_output_df_2["time"][0] == "1548772230"
    assert test_output_df_2["search_name"][0] == "Test Search Name 2"
    assert test_output_df_2["orig_time"][0] == ""
    assert test_output_df_2["urgency"][0] == ""
    assert test_output_df_2["user"][0] == ""
    assert test_output_df_2["owner"][0] == ""
    assert test_output_df_2["security_domain"][0] == ""
    assert test_output_df_2["severity"][0] == ""
    assert test_output_df_2["src_ip"][0] == "10.01.01.123"
    assert test_output_df_2["src_mac"][0] == "1a:2b:3c:4d:5e:6f"
    assert test_output_df_2["src_port"][0] == ""
    # dest_ip is obtained from dest attribute. Since data doesn't have dest_ip.
    assert test_output_df_2["dest_ip"][0] == "10.01.01.122"
    assert test_output_df_2["dest_mac"][0] == "1a:2b:3c:4d:5e:7g"
    assert test_output_df_2["dest_port"][0] == ""
    assert test_output_df_2["dest_priority"][0] == ""
    assert test_output_df_2["device_name"][0] == ""
    assert test_output_df_2["event_name"][0] == ""
    assert test_output_df_2["event_type"][0] == ""
    assert test_output_df_2["ip_address"][0] == ""
    assert test_output_df_2["message_ip"][0] == ""
    assert test_output_df_2["message_username"][0] == ""
    assert test_output_df_2["message_hostname"][0] == ""
    assert test_output_df_2["message_description"][0] == ""

    test_input_3 = cudf.Series([TEST_DATA3])
    test_output_df_3 = snp.parse(test_input_3)
    assert len(test_output_df_3.columns) == 23
    assert test_output_df_3["time"][0] == "1548234811"
    assert test_output_df_3["search_name"][0] == "Test Search Name 3"
    assert test_output_df_3["orig_time"][0] == ""
    assert test_output_df_3["urgency"][0] == ""
    assert test_output_df_3["user"][0] == ""
    assert test_output_df_3["owner"][0] == ""
    assert test_output_df_3["security_domain"][0] == ""
    assert test_output_df_3["severity"][0] == ""
    # src_ip is obtained from src attribute. Since data doesn't have src_ip.
    assert test_output_df_3["src_ip"][0] == "10.01.01.123"
    assert test_output_df_3["src_mac"][0] == "1a:2b:3c:4d:5e:6f"
    assert test_output_df_3["src_port"][0] == ""
    # dest_ip is obtained from dest attribute. Since data doesn't have dest_ip.
    assert test_output_df_3["dest_ip"][0] == "10.01.01.122"
    assert test_output_df_3["dest_mac"][0] == "1a:2b:3c:4d:5e:7g"
    assert test_output_df_3["dest_port"][0] == ""
    assert test_output_df_3["dest_priority"][0] == ""
    assert test_output_df_3["device_name"][0] == ""
    assert test_output_df_3["event_name"][0] == ""
    assert test_output_df_3["event_type"][0] == ""
    assert test_output_df_3["ip_address"][0] == ""
    assert test_output_df_3["message_ip"][0] == ""
    assert test_output_df_3["message_username"][0] == ""
    assert test_output_df_3["message_hostname"][0] == ""
    assert test_output_df_3["message_description"][0] == ""

    test_input_4 = cudf.Series([TEST_DATA4])
    test_output_df_4 = snp.parse(test_input_4)
    assert len(test_output_df_4.columns) == 23
    assert test_output_df_4["time"][0] == "1566345700"
    assert (test_output_df_4["search_name"][0] == "Endpoint - Brute Force against Known User - Rule")
    assert test_output_df_4["orig_time"][0] == ""
    assert test_output_df_4["urgency"][0] == ""
    assert test_output_df_4["user"][0] == "pjame"
    assert test_output_df_4["owner"][0] == ""
    assert test_output_df_4["security_domain"][0] == ""
    assert test_output_df_4["severity"][0] == ""
    assert test_output_df_4["src_ip"][0] == ""
    assert test_output_df_4["src_mac"][0] == ""
    assert test_output_df_4["src_port"][0] == ""
    assert test_output_df_4["dest_ip"][0] == ""
    assert test_output_df_4["dest_mac"][0] == ""
    assert test_output_df_4["dest_port"][0] == ""
    assert test_output_df_4["dest_priority"][0] == ""
    assert test_output_df_4["device_name"][0] == ""
    assert test_output_df_4["event_name"][0] == ""
    assert test_output_df_4["event_type"][0] == ""
    assert test_output_df_4["ip_address"][0] == ""
    assert test_output_df_4["message_ip"][0] == ""
    assert test_output_df_4["message_username"][0] == ""
    assert test_output_df_4["message_hostname"][0] == ""
    assert test_output_df_4["message_description"][0] == ""

    test_input_5 = cudf.Series([TEST_DATA5])
    test_output_df_5 = snp.parse(test_input_5)
    assert len(test_output_df_5.columns) == 23
    assert test_output_df_5["time"][0] == "1566345700"
    assert test_output_df_5["search_name"][0] == "Manual Notable Event - Rule"
    assert test_output_df_5["orig_time"][0] == ""
    assert test_output_df_5["urgency"][0] == "medium"
    assert test_output_df_5["user"][0] == ""
    assert test_output_df_5["owner"][0] == "test@nvidia.com"
    assert test_output_df_5["security_domain"][0] == "endpoint"
    assert test_output_df_5["severity"][0] == ""
    assert test_output_df_5["src_ip"][0] == ""
    assert test_output_df_5["src_mac"][0] == ""
    assert test_output_df_5["src_port"][0] == ""
    assert test_output_df_5["dest_ip"][0] == ""
    assert test_output_df_5["dest_mac"][0] == ""
    assert test_output_df_5["dest_port"][0] == ""
    assert test_output_df_5["dest_priority"][0] == ""
    assert test_output_df_5["device_name"][0] == ""
    assert test_output_df_5["event_name"][0] == ""
    assert test_output_df_5["event_type"][0] == ""
    assert test_output_df_5["ip_address"][0] == ""
    assert test_output_df_5["message_ip"][0] == ""
    assert test_output_df_5["message_username"][0] == ""
    assert test_output_df_5["message_hostname"][0] == ""
    assert test_output_df_5["message_description"][0] == ""

    test_input_6 = cudf.Series([TEST_DATA6])
    test_output_df_6 = snp.parse(test_input_6)
    assert len(test_output_df_6.columns) == 23
    assert test_output_df_6["time"][0] == "1566345700"
    assert (test_output_df_6["search_name"][0] == "Endpoint - FireEye NX alert for Incident Review (Minor) - Rule")
    assert test_output_df_6["orig_time"][0] == ""
    assert test_output_df_6["urgency"][0] == ""
    assert test_output_df_6["user"][0] == ""
    assert test_output_df_6["owner"][0] == ""
    assert test_output_df_6["security_domain"][0] == ""
    assert test_output_df_6["severity"][0] == "minr"
    assert test_output_df_6["src_ip"][0] == "10.15.90.151"
    assert test_output_df_6["src_mac"][0] == ""
    assert test_output_df_6["src_port"][0] == "40472"
    assert test_output_df_6["dest_ip"][0] == "10.15.90.150"
    assert test_output_df_6["dest_mac"][0] == ""
    assert test_output_df_6["dest_port"][0] == "80"
    assert test_output_df_6["dest_priority"][0] == ""
    assert test_output_df_6["device_name"][0] == ""
    assert test_output_df_6["event_name"][0] == ""
    assert test_output_df_6["event_type"][0] == ""
    assert test_output_df_6["ip_address"][0] == ""
    assert test_output_df_6["message_ip"][0] == ""
    assert test_output_df_6["message_username"][0] == ""
    assert test_output_df_6["message_hostname"][0] == ""
    assert test_output_df_6["message_description"][0] == ""

    test_input_7 = cudf.Series([TEST_DATA7])
    test_output_df7 = snp.parse(test_input_7)
    assert len(test_output_df7.columns) == 23
    assert test_output_df7["time"][0] == "1566345700"
    assert (test_output_df7["search_name"][0] == "Endpoint - Host With Malware Detected (Quarantined or Waived) - Rule")
    assert test_output_df7["orig_time"][0] == ""
    assert test_output_df7["urgency"][0] == ""
    assert test_output_df7["user"][0] == ""
    assert test_output_df7["owner"][0] == ""
    assert test_output_df7["security_domain"][0] == ""
    assert test_output_df7["severity"][0] == ""
    assert test_output_df7["src_ip"][0] == ""
    assert test_output_df7["src_mac"][0] == ""
    assert test_output_df7["src_port"][0] == ""
    assert test_output_df7["dest_ip"][0] == "TEST-01"
    assert test_output_df7["dest_mac"][0] == ""
    assert test_output_df7["dest_port"][0] == ""
    assert test_output_df7["dest_priority"][0] == "medium"
    assert test_output_df7["device_name"][0] == ""
    assert test_output_df7["event_name"][0] == ""
    assert test_output_df7["event_type"][0] == ""
    assert test_output_df7["ip_address"][0] == ""
    assert test_output_df7["message_ip"][0] == ""
    assert test_output_df7["message_username"][0] == ""
    assert test_output_df7["message_hostname"][0] == ""
    assert test_output_df7["message_description"][0] == ""
