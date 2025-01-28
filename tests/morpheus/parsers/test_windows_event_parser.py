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

import os

import pytest

import cudf

from _utils import TEST_DIRS
from morpheus.parsers.windows_event_parser import WindowsEventParser


def validate_4624(parsed_rec):
    assert parsed_rec["time"] == "04/01/2019 07:07:21 pm"
    assert parsed_rec["id"] == "c54d7f17-8eb8-4d78-a8f7-4b681256e2b3"
    assert parsed_rec["eventcode"] == "4624"
    assert (parsed_rec["detailed_authentication_information_authentication_package"] == "kerberos")
    assert (parsed_rec["new_logon_logon_guid"] == "{e53069f0-662e-0c65-f889-aa8d8770d56a}")
    assert parsed_rec["failure_information_failure_reason"] == ""
    assert parsed_rec["failure_information_status"] == ""
    assert parsed_rec["computername"] == ""
    assert parsed_rec["new_logon_logon_id"] == "0x9de8990de"
    assert parsed_rec["subject_security_id"] == "null sid"
    assert (parsed_rec["detailed_authentication_information_package_name_ntlm_only"] == "-")
    assert parsed_rec["logon_type"] == "3"
    assert parsed_rec["account_for_which_logon_failed_security_id"] == ""
    assert parsed_rec["detailed_authentication_information_key_length"] == "0"
    assert parsed_rec["subject_logon_id"] == "0x0"
    assert parsed_rec["process_information_caller_process_name"] == ""
    assert parsed_rec["process_information_caller_process_id"] == ""
    assert parsed_rec["subject_account_name"] == "-"
    assert parsed_rec["process_information_process_name"] == "-"
    assert parsed_rec["new_logon_account_name"] == "test106$"
    assert parsed_rec["process_information_process_id"] == "0x0"
    assert parsed_rec["failure_information_sub_status"] == ""
    assert parsed_rec["new_logon_security_id"] == "test.comest"
    assert parsed_rec["network_information_source_network_address"] == "100.00.100.1"
    assert parsed_rec["detailed_authentication_information_transited_services"] == "-"
    assert parsed_rec["new_logon_account_domain"] == "test.com"
    assert parsed_rec["subject_account_domain"] == "-"
    assert parsed_rec["detailed_authentication_information_logon_process"] == "kerberos"
    assert parsed_rec["account_for_which_logon_failed_account_domain"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_name"] == ""
    assert parsed_rec["network_information_workstation_name"] == ""
    assert parsed_rec["network_information_source_port"] == "39028"
    assert parsed_rec["application_information_process_id"] == ""
    assert parsed_rec["application_information_application_name"] == ""
    assert parsed_rec["network_information_direction"] == ""
    assert parsed_rec["network_information_source_address"] == ""
    assert parsed_rec["network_information_destination_address"] == ""
    assert parsed_rec["network_information_destination_port"] == ""
    assert parsed_rec["network_information_protocol"] == ""
    assert parsed_rec["filter_information_filter_run_time_id"] == ""
    assert parsed_rec["filter_information_layer_name"] == ""
    assert parsed_rec["filter_information_layer_run_time_id"] == ""


def validate_4625(parsed_rec):
    assert parsed_rec["time"] == "04/03/2019 05:57:33 am"
    assert parsed_rec["id"] == "cf4876f3-716c-415c-994e-84acda054c9c"
    assert parsed_rec["eventcode"] == "4625"
    assert (parsed_rec["detailed_authentication_information_authentication_package"] == "ntlm")
    assert parsed_rec["new_logon_logon_guid"] == ""
    assert (parsed_rec["failure_information_failure_reason"] == "unknown user name or bad password.")
    assert parsed_rec["failure_information_status"] == "0xc000006d"
    assert parsed_rec["computername"] == "abc.test.com"
    assert parsed_rec["new_logon_logon_id"] == ""
    assert parsed_rec["subject_security_id"] == "null sid"
    assert (parsed_rec["detailed_authentication_information_package_name_ntlm_only"] == "-")
    assert parsed_rec["logon_type"] == "3"
    assert parsed_rec["account_for_which_logon_failed_security_id"] == "null sid"
    assert parsed_rec["detailed_authentication_information_key_length"] == "0"
    assert parsed_rec["subject_logon_id"] == "0x0"
    assert parsed_rec["process_information_caller_process_name"] == "-"
    assert parsed_rec["process_information_caller_process_id"] == "0x0"
    assert parsed_rec["subject_account_name"] == "-"
    assert parsed_rec["process_information_process_name"] == ""
    assert parsed_rec["new_logon_account_name"] == ""
    assert parsed_rec["process_information_process_id"] == ""
    assert parsed_rec["failure_information_sub_status"] == "0xc0000064"
    assert parsed_rec["new_logon_security_id"] == ""
    assert parsed_rec["network_information_source_network_address"] == "10.10.100.20"
    assert parsed_rec["detailed_authentication_information_transited_services"] == "-"
    assert parsed_rec["new_logon_account_domain"] == ""
    assert parsed_rec["subject_account_domain"] == "-"
    assert parsed_rec["detailed_authentication_information_logon_process"] == "ntlmssp"
    assert parsed_rec["account_for_which_logon_failed_account_domain"] == "hxyz"
    assert parsed_rec["account_for_which_logon_failed_account_name"] == "hxyz"
    assert parsed_rec["network_information_workstation_name"] == "hxyz-pc1"
    assert parsed_rec["network_information_source_port"] == "53662"
    assert parsed_rec["application_information_process_id"] == ""
    assert parsed_rec["application_information_application_name"] == ""
    assert parsed_rec["network_information_direction"] == ""
    assert parsed_rec["network_information_source_address"] == ""
    assert parsed_rec["network_information_destination_address"] == ""
    assert parsed_rec["network_information_destination_port"] == ""
    assert parsed_rec["network_information_protocol"] == ""
    assert parsed_rec["filter_information_filter_run_time_id"] == ""
    assert parsed_rec["filter_information_layer_name"] == ""
    assert parsed_rec["filter_information_layer_run_time_id"] == ""


def validate_5156(parsed_rec):
    assert parsed_rec["time"] == "04/03/2019 11:58:59 am"
    assert parsed_rec["id"] == "c3f48bba-90a1-4999-84a6-4da9d964d31d"
    assert parsed_rec["eventcode"] == "5156"
    assert (parsed_rec["detailed_authentication_information_authentication_package"] == "")
    assert parsed_rec["new_logon_logon_guid"] == ""
    assert parsed_rec["failure_information_failure_reason"] == ""
    assert parsed_rec["failure_information_status"] == ""
    assert parsed_rec["computername"] == ""
    assert parsed_rec["new_logon_logon_id"] == ""
    assert parsed_rec["subject_security_id"] == ""
    assert (parsed_rec["detailed_authentication_information_package_name_ntlm_only"] == "")
    assert parsed_rec["logon_type"] == ""
    assert parsed_rec["account_for_which_logon_failed_security_id"] == ""
    assert parsed_rec["detailed_authentication_information_key_length"] == ""
    assert parsed_rec["subject_logon_id"] == ""
    assert parsed_rec["process_information_caller_process_name"] == ""
    assert parsed_rec["process_information_caller_process_id"] == ""
    assert parsed_rec["subject_account_name"] == ""
    assert parsed_rec["process_information_process_name"] == ""
    assert parsed_rec["new_logon_account_name"] == ""
    assert parsed_rec["process_information_process_id"] == ""
    assert parsed_rec["failure_information_sub_status"] == ""
    assert parsed_rec["new_logon_security_id"] == ""
    assert parsed_rec["network_information_source_network_address"] == ""
    assert parsed_rec["detailed_authentication_information_transited_services"] == ""
    assert parsed_rec["new_logon_account_domain"] == ""
    assert parsed_rec["subject_account_domain"] == ""
    assert parsed_rec["detailed_authentication_information_logon_process"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_domain"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_name"] == ""
    assert parsed_rec["network_information_workstation_name"] == ""
    assert parsed_rec["network_information_source_port"] == "138"
    assert parsed_rec["application_information_process_id"] == "4"
    assert parsed_rec["application_information_application_name"] == "system"
    assert parsed_rec["network_information_direction"] == "inbound"
    assert parsed_rec["network_information_source_address"] == "100.20.100.20"
    assert parsed_rec["network_information_destination_address"] == "100.20.100.30"
    assert parsed_rec["network_information_destination_port"] == "138"
    assert parsed_rec["network_information_protocol"] == "17"
    assert parsed_rec["filter_information_filter_run_time_id"] == "0"
    assert parsed_rec["filter_information_layer_name"] == "receive/accept"
    assert parsed_rec["filter_information_layer_run_time_id"] == "44"


def validate_5157(parsed_rec):
    assert parsed_rec["time"] == "04/03/2019 11:58:59 am"
    assert parsed_rec["id"] == "565beda9-346a-46a3-9f1f-25eab8d3414d"
    assert parsed_rec["eventcode"] == "5157"
    assert (parsed_rec["detailed_authentication_information_authentication_package"] == "")
    assert parsed_rec["new_logon_logon_guid"] == ""
    assert parsed_rec["failure_information_failure_reason"] == ""
    assert parsed_rec["failure_information_status"] == ""
    assert parsed_rec["computername"] == ""
    assert parsed_rec["new_logon_logon_id"] == ""
    assert parsed_rec["subject_security_id"] == ""
    assert (parsed_rec["detailed_authentication_information_package_name_ntlm_only"] == "")
    assert parsed_rec["logon_type"] == ""
    assert parsed_rec["account_for_which_logon_failed_security_id"] == ""
    assert parsed_rec["detailed_authentication_information_key_length"] == ""
    assert parsed_rec["subject_logon_id"] == ""
    assert parsed_rec["process_information_caller_process_name"] == ""
    assert parsed_rec["process_information_caller_process_id"] == ""
    assert parsed_rec["subject_account_name"] == ""
    assert parsed_rec["process_information_process_name"] == ""
    assert parsed_rec["new_logon_account_name"] == ""
    assert parsed_rec["process_information_process_id"] == ""
    assert parsed_rec["failure_information_sub_status"] == ""
    assert parsed_rec["new_logon_security_id"] == ""
    assert parsed_rec["network_information_source_network_address"] == ""
    assert parsed_rec["detailed_authentication_information_transited_services"] == ""
    assert parsed_rec["new_logon_account_domain"] == ""
    assert parsed_rec["subject_account_domain"] == ""
    assert parsed_rec["detailed_authentication_information_logon_process"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_domain"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_name"] == ""
    assert parsed_rec["network_information_workstation_name"] == ""
    assert parsed_rec["network_information_source_port"] == "137"
    assert parsed_rec["application_information_process_id"] == "1048"
    assert (parsed_rec["application_information_application_name"] ==
            "\\device\\harddiskvolume1\\windows\\system32\\svchost.exe")
    assert parsed_rec["network_information_direction"] == "inbound"
    assert parsed_rec["network_information_source_address"] == "100.20.100.30"
    assert parsed_rec["network_information_destination_address"] == "100.20.100.20"
    assert parsed_rec["network_information_destination_port"] == "137"
    assert parsed_rec["network_information_protocol"] == "0"
    assert parsed_rec["filter_information_filter_run_time_id"] == "65595"
    assert parsed_rec["filter_information_layer_name"] == "receive/accept"
    assert parsed_rec["filter_information_layer_run_time_id"] == "44"


def validate_4798(parsed_rec):
    assert parsed_rec["time"] == "04/03/2019 05:57:33 am"
    assert parsed_rec["id"] == "cf4876f3-716c-415c-994e-84acda054c9c"
    assert parsed_rec["eventcode"] == "4798"
    assert parsed_rec["subject_security_id"] == "null sid"
    assert parsed_rec["subject_account_name"] == ""
    assert parsed_rec["subject_logon_id"] == "0x0"
    assert parsed_rec["user_security_id"] == "null sid"
    assert parsed_rec["user_account_name"] == "hxyz"
    assert parsed_rec["user_account_domain"] == "hxyz-pc1"
    assert parsed_rec["process_information_process_id"] == "0x0"
    assert parsed_rec["process_information_process_name"] == "-"


def validate_4769(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 04:45:36 am"
    assert parsed_rec["id"] == "cf4876f3-716c-415c-994e-84acda054c9c"
    assert parsed_rec["eventcode"] == "4769"
    assert parsed_rec["account_information_account_name"] == "user@localhost.com"
    assert parsed_rec["account_information_account_domain"] == "localhost.com"
    assert (parsed_rec["account_information_logon_guid"] == "{1f1d4c09-e154-4898-4eb8-e3a03e130d11}")
    assert parsed_rec["service_information_service_name"] == "test.localhost.com"
    assert parsed_rec["service_information_service_id"] == "none_mapped"
    assert parsed_rec["network_information_client_address"] == "::ffff:100.10.100.20"
    assert parsed_rec["network_information_client_port"] == "26061"
    assert parsed_rec["additional_information_ticket_options"] == "0x40810000"
    assert parsed_rec["additional_information_ticket_encryption_type"] == "0x17"
    assert parsed_rec["additional_information_failure_code"] == "0x0"
    assert parsed_rec["additional_information_transited_services"] == "-"


def validate_4770(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4770"
    assert parsed_rec["account_information_account_name"] == "test@localhost.com"
    assert parsed_rec["account_information_account_domain"] == "localhost.com"
    assert parsed_rec["service_information_service_name"] == "user"
    assert parsed_rec["service_information_service_id"] == "localhost"
    assert parsed_rec["network_information_client_address"] == "::ffff:10.30.100.130"
    assert parsed_rec["network_information_client_port"] == "62133"
    assert parsed_rec["additional_information_ticket_options"] == "0x50800002"
    assert parsed_rec["additional_information_ticket_encryption_type"] == "0x12"


def validate_4771(parsed_rec):
    assert parsed_rec["time"] == "12/06/2018 06:52:05 am"
    assert parsed_rec["id"] == "cf4876f3-716c-415c-994e-84acda054c9c"
    assert parsed_rec["eventcode"] == "4771"
    assert parsed_rec["account_information_security_id"] == "localhost.com\\lab"
    assert parsed_rec["account_information_account_name"] == "lab"
    assert parsed_rec["service_information_service_name"] == "user/localhost.com"
    assert parsed_rec["network_information_client_address"] == "100.20.1.70"
    assert parsed_rec["network_information_client_port"] == "60284"
    assert parsed_rec["additional_information_ticket_options"] == "0x40800000"
    assert parsed_rec["additional_information_failure_code"] == "0x18"
    assert parsed_rec["additional_information_pre_authentication_type"] == "2"
    assert parsed_rec["certificate_information_certificate_issuer_name"] == ""
    assert parsed_rec["certificate_information_certificate_serial_number"] == ""
    assert parsed_rec["certificate_information_certificate_thumbprint"] == ""


def validate_4781(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4781"
    assert parsed_rec["subject_security_id"] == "acme\\administrator"
    assert parsed_rec["subject_account_domain"] == "localhost.com"
    assert parsed_rec["subject_account_name"] == "test@localhost.com"
    assert parsed_rec["subject_logon_id"] == "0x1f40f"
    assert parsed_rec["target_account_security_id"] == "acme\\emp-nbonaparte"
    assert parsed_rec["target_account_account_domain"] == "acme"
    assert parsed_rec["target_account_old_account_name"] == "nbonaparte"
    assert parsed_rec["target_account_new_account_name"] == "emp-nbonaparte"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4782(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4782"
    assert parsed_rec["subject_security_id"] == "acme\\administrator"
    assert parsed_rec["subject_account_domain"] == "localhost.com"
    assert parsed_rec["subject_account_name"] == "test@localhost.com"
    assert parsed_rec["subject_logon_id"] == "0x1f40f"
    assert parsed_rec["target_account_account_domain"] == "acme"
    assert parsed_rec["target_account_account_name"] == "nbonaparte"


def validate_4647(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4647"
    assert parsed_rec["subject_security_id"] == "anonymous logon"
    assert parsed_rec["subject_account_name"] == "appservice"
    assert parsed_rec["subject_account_domain"] == "domain001"
    assert parsed_rec["subject_logon_id"] == "0x27b9013"


def validate_4634(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4634"
    assert parsed_rec["subject_security_id"] == "anonymous logon"
    assert parsed_rec["subject_account_name"] == "appservice"
    assert parsed_rec["subject_account_domain"] == "domain001"
    assert parsed_rec["subject_logon_id"] == "0x27b9013"
    assert parsed_rec["logon_type"] == "3"


def validate_4648(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4648"
    assert parsed_rec["subject_account_name"] == "administrator"
    assert parsed_rec["subject_account_domain"] == "win-r9h529rio4y"
    assert parsed_rec["subject_logon_id"] == "0x1ba0e"
    assert parsed_rec["subject_logon_guid"] == "{00000000-0000-0000-0000-000000000000}"
    assert (parsed_rec["account_whose_credentials_were_used_account_name"] == "rsmith@mtg.com")
    assert (parsed_rec["account_whose_credentials_were_used_account_domain"] == "win-r9h529rio4y")
    assert (parsed_rec["account_whose_credentials_were_used_logon_guid"] == "{00000000-0000-0000-0000-000000000000}")
    assert parsed_rec["target_server_target_server_name"] == "sp01.icemail.com"
    assert parsed_rec["target_server_additional_information"] == "sp01.icemail.com"
    assert parsed_rec["process_information_process_id"] == "0x77c"
    assert (parsed_rec["process_information_process_name"] == "c:\\program files\\internet explorer\\iexplore.exe")
    assert parsed_rec["network_information_network_address"] == "-"
    assert parsed_rec["network_information_port"] == "-"


def validate_4672(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:52:50 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4672"
    assert parsed_rec["time"] == "09/27/2018 10:52:50 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4672"
    assert parsed_rec["subject_security_id"] == "devuser"
    assert parsed_rec["subject_account_name"] == "user"
    assert parsed_rec["subject_account_domain"] == "dev"
    assert parsed_rec["subject_logon_id"] == "0x800a513d"
    assert (parsed_rec["privileges"] ==
            "sesecurityprivilege|sebackupprivilege|serestoreprivilege|setakeownershipprivilege|sedebugprivilege|"
            "sesystemenvironmentprivilege|seloaddriverprivilege|seimpersonateprivilege")


def validate_4673(parsed_rec):
    assert parsed_rec["time"] == "04/30/2018 05:13:59 pm"
    assert parsed_rec["id"] == "sdgfhsdfhj-3245-dsf"
    assert parsed_rec["eventcode"] == "4673"
    assert parsed_rec["subject_security_id"] == "nt authority\\system"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "fvjbvfjbvf$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert (parsed_rec["service_server"] == "nt local security authority / authentication service")
    assert parsed_rec["service_service_name"] == "lsaregisterlogonprocess()"
    assert parsed_rec["process_process_id"] == "0x234"
    assert parsed_rec["process_process_name"] == "c:\\windows\\system32\\lsass.exe"
    assert parsed_rec["privileges"] == "setcbprivilege"


def validate_4722(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 09:56:10 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4722"
    assert parsed_rec["subject_security_id"] == "test.com\\dhgfckkcg"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "dhgfckkcg"
    assert parsed_rec["subject_logon_id"] == "0x2d55e5ef7"
    assert parsed_rec["target_account_security_id"] == "test.com\\hgcghjj"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "hgcghjj"


def validate_4720(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 09:56:10 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4720"
    assert parsed_rec["subject_security_id"] == "acme-fr\administrator"
    assert parsed_rec["subject_account_domain"] == "acme-fr"
    assert parsed_rec["subject_account_name"] == "administrator"
    assert parsed_rec["subject_logon_id"] == "0x20f9d"
    assert parsed_rec["new_account_security_id"] == "acme-fr\\john.lockeaccount"
    assert parsed_rec["new_account_account_name"] == "john.locke"
    assert parsed_rec["new_account_domain_name"] == "acme-fr"
    assert parsed_rec["attributes_sam_account_name"] == "john.locke"
    assert parsed_rec["attributes_display_name"] == "john locke"
    assert parsed_rec["attributes_user_principal_name"] == "john.locke@acme-fr.local"
    assert parsed_rec["attributes_home_directory"] == "-"
    assert parsed_rec["attributes_home_drive"] == "-"
    assert parsed_rec["attributes_script_path"] == "-"
    assert parsed_rec["attributes_profile_path"] == "-"
    assert parsed_rec["attributes_user_workstations"] == "-"
    assert parsed_rec["attributes_password_last_set"] == "<never>"
    assert parsed_rec["attributes_account_expires"] == "<never>"
    assert parsed_rec["attributes_primary_group_id"] == "513"
    assert parsed_rec["attributes_allowed_to_delegate_to"] == "-"
    assert parsed_rec["attributes_old_uac_value"] == "0x0"
    assert parsed_rec["attributes_new_uac_value"] == "0x15"
    assert (parsed_rec["attributes_user_account_control"] ==
            "account disabled|'password not required' - enabled|'normal account' - enable")
    assert parsed_rec["attributes_user_parameters"] == "-"
    assert parsed_rec["attributes_sid_history"] == "-"
    assert parsed_rec["attributes_logon_hours"] == "<value not set>"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4723(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4723"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4724(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4724"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"


def validate_4725(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4725"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"


def validate_4726(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4726"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4732(parsed_rec):
    assert parsed_rec["time"] == "09/19/2018 06:18:24 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4732"
    assert parsed_rec["subject_security_id"] == "nt authority\\system"
    assert parsed_rec["subject_account_domain"] == "localhost.com"
    assert parsed_rec["subject_account_name"] == "testuser$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert parsed_rec["member_security_id"] == "testuser\\offer"
    assert parsed_rec["member_account_name"] == "-"
    assert parsed_rec["group_security_id"] == "testuser\\offer remote assistance helpers"
    assert parsed_rec["group_group_name"] == "offer remote assistance helpers"
    assert parsed_rec["group_group_domain"] == "testuser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4738(parsed_rec):
    assert parsed_rec["time"] == "05/01/2018 05:41:37 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4738"
    assert parsed_rec["subject_security_id"] == "nt authority\\system"
    assert parsed_rec["subject_account_domain"] == "prod"
    assert parsed_rec["subject_account_name"] == "esfdhf06$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert parsed_rec["target_account_security_id"] == "esfdhf06\\atble"
    assert parsed_rec["target_account_account_domain"] == "esfdhf06"
    assert parsed_rec["target_account_account_name"] == "atble"
    assert parsed_rec["additional_information_privileges"] == "-"
    assert parsed_rec["changed_attributes_sam_account_name"] == "atble"
    assert parsed_rec["changed_attributes_home_directory"] == "<value not set>"
    assert parsed_rec["changed_attributes_primary_group_id"] == "513"
    assert parsed_rec["changed_attributes_user_principal_name"] == "-"
    assert parsed_rec["changed_attributes_profile_path"] == "<value not set>"
    assert parsed_rec["changed_attributes_user_workstations"] == "<value not set>"
    assert parsed_rec["changed_attributes_user_parameters"] == "-"
    assert parsed_rec["changed_attributes_script_path"] == "<value not set>"
    assert parsed_rec["changed_attributes_display_name"] == "mike atble"
    assert parsed_rec["changed_attributes_home_drive"] == "<value not set>"
    assert parsed_rec["changed_attributes_new_uac_value"] == "0x210"
    assert parsed_rec["changed_attributes_logon_hours"] == "all"
    assert parsed_rec["changed_attributes_account_expires"] == "<never>"
    assert parsed_rec["changed_attributes_old_uac_value"] == "0x210"
    assert parsed_rec["changed_attributes_password_last_set"] == "5/1/2018 5:41:37 am"
    assert parsed_rec["changed_attributes_allowedtodelegateto"] == "-"
    assert parsed_rec["changed_attributes_user_account_control"] == "-"
    assert parsed_rec["changed_attributes_sid_history"] == "-"


def validate_4740(parsed_rec):
    assert parsed_rec["time"] == "09/28/2018 01:53:37 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4740"
    assert parsed_rec["subject_security_id"] == "nt authority\\system"
    assert parsed_rec["subject_account_domain"] == "nvdmz"
    assert parsed_rec["subject_account_name"] == "sdgbjsd02$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert parsed_rec["account_locked_out_security_id"] == "sdgbjsd02\\guest"
    assert parsed_rec["account_locked_out_account_name"] == "guest"
    assert parsed_rec["additional_information_caller_computer_name"] == "sdgbjsd01"


def validate_4743(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4743"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4756(parsed_rec):
    assert parsed_rec["time"] == "09/19/2018 06:18:24 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4756"
    assert parsed_rec["subject_security_id"] == "nt authority\\system"
    assert parsed_rec["subject_account_domain"] == "localhost.com"
    assert parsed_rec["subject_account_name"] == "testuser$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert parsed_rec["member_security_id"] == "testuser\\offer"
    assert parsed_rec["member_account_name"] == "-"
    assert parsed_rec["group_security_id"] == "testuser\\offer remote assistance helpers"
    assert parsed_rec["group_group_name"] == "offer remote assistance helpers"
    assert parsed_rec["group_group_domain"] == "testuser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4767(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4767"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"


def validate_4768(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 09:08:02 am"
    assert parsed_rec["id"] == "asd-eter-34235-fgd-346"
    assert parsed_rec["eventcode"] == "4768"
    assert parsed_rec["network_information_client_address"] == "::ffff:10.20.90.30"
    assert parsed_rec["network_information_client_port"] == "6349"
    assert parsed_rec["service_information_service_name"] == "asdfgrvk"
    assert parsed_rec["service_information_service_id"] == "localhost.com\\asdfgrvk"
    assert parsed_rec["account_information_account_name"] == "healthmailbox06ca30c"
    assert parsed_rec["account_information_supplied_realm_name"] == "localhost.com"
    assert (parsed_rec["account_information_user_id"] == "localhost.com\\healthmailbox06ca30c")
    assert parsed_rec["additional_information_result_code"] == "0x0"
    assert parsed_rec["additional_information_ticket_options"] == "0x40810010"
    assert parsed_rec["additional_information_ticket_encryption_type"] == "0x12"
    assert parsed_rec["additional_information_pre_authentication_type"] == "2"
    assert parsed_rec["certificate_information_certificate_issuer_name"] == ""
    assert parsed_rec["certificate_information_certificate_serial_number"] == ""
    assert parsed_rec["certificate_information_certificate_thumbprint"] == ""


def unknown_record_type(parsed_rec):
    raise ValueError("Unknown eventcode appeared")


VALIDATE_DICT = {
    "4624": validate_4624,
    "4625": validate_4625,
    "4634": validate_4634,
    "4647": validate_4647,
    "4648": validate_4648,
    "4672": validate_4672,
    "4673": validate_4673,
    "4720": validate_4720,
    "4722": validate_4722,
    "4723": validate_4723,
    "4724": validate_4724,
    "4725": validate_4725,
    "4726": validate_4726,
    "4732": validate_4732,
    "4738": validate_4738,
    "4740": validate_4740,
    "4743": validate_4743,
    "4756": validate_4756,
    "4767": validate_4767,
    "4768": validate_4768,
    "4769": validate_4769,
    "4770": validate_4770,
    "4771": validate_4771,
    "4781": validate_4781,
    "4782": validate_4782,
    "4798": validate_4798,
    "5156": validate_5156,
    "5157": validate_5157,
}


def test_windows_event_parser():
    wep = WindowsEventParser()

    with open(os.path.join(TEST_DIRS.tests_data_dir, 'windows_event_logs.txt'), encoding='UTF-8') as fh:
        test_logs = fh.readlines()
    test_input = cudf.Series(test_logs)
    test_output_df = wep.parse(test_input)

    for parsed_rec in test_output_df.to_records():
        eventcode = parsed_rec["eventcode"]
        validate_func = VALIDATE_DICT.get(eventcode, unknown_record_type)
        validate_func(parsed_rec)


def test2_windows_event_parser():
    wep = WindowsEventParser(interested_eventcodes=["5156"])
    with open(os.path.join(TEST_DIRS.tests_data_dir, 'windows_event_logs.txt'), encoding='UTF-8') as fh:
        test_logs = fh.readlines()
    test_input = cudf.Series(test_logs)
    test_output_df = wep.parse(test_input)
    parsed_rec = test_output_df.to_records()[0]
    assert parsed_rec["time"] == "04/03/2019 11:58:59 am"
    assert parsed_rec["id"] == "c3f48bba-90a1-4999-84a6-4da9d964d31d"
    assert parsed_rec["eventcode"] == "5156"
    assert parsed_rec["application_information_process_id"] == "4"
    assert parsed_rec["application_information_application_name"] == "system"
    assert parsed_rec["network_information_direction"] == "inbound"
    assert parsed_rec["network_information_source_address"] == "100.20.100.20"
    assert parsed_rec["network_information_source_port"] == "138"
    assert parsed_rec["network_information_destination_address"] == "100.20.100.30"
    assert parsed_rec["network_information_destination_port"] == "138"
    assert parsed_rec["network_information_protocol"] == "17"
    assert parsed_rec["filter_information_filter_run_time_id"] == "0"
    assert parsed_rec["filter_information_layer_name"] == "receive/accept"
    assert parsed_rec["filter_information_layer_run_time_id"] == "44"


def test3_windows_event_parser():
    expected_error = KeyError(
        "Regex for eventcode 24 is not available in the config file. Please choose from ['4624', '4625', "
        "'4634', '4647', '4648', '4672', '4673', '4720', '4722', '4723', '4724', '4725', '4726', '4732', '4738', "
        "'4740', '4743', '4756', '4767', '4768', '4769', '4770', '4771', '4781', '4782', '4798', '5156', '5157']")
    with pytest.raises(KeyError) as actual_error:
        WindowsEventParser(interested_eventcodes=["5156", "24"])
        assert actual_error == expected_error
