"""unitree_go::msg::dds_::ConfigChangeStatus_ {string name; string content;}.

Missing from unitree_sdk2_python; layout taken from the official C++ header
(unitree/idl/go2/ConfigChangeStatus_.hpp). Read-only use (subscriber).
"""
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate


@dataclass
@annotate.final
@annotate.autoid("sequential")
class ConfigChangeStatus_(idl.IdlStruct, typename="unitree_go.msg.dds_.ConfigChangeStatus_"):
    name: str
    content: str
