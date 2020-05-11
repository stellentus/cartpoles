# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: remote.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='remote.proto',
  package='',
  syntax='proto3',
  serialized_options=b'Z\010.;remote',
  serialized_pb=b'\n\x0cremote.proto\" \n\nAttributes\x12\x12\n\nattributes\x18\x01 \x01(\t\"T\n\x0f\x41gentAttributes\x12\x1f\n\nexperiment\x18\x01 \x01(\x0b\x32\x0b.Attributes\x12 \n\x0b\x65nvironment\x18\x02 \x01(\x0b\x32\x0b.Attributes\"\x17\n\x05State\x12\x0e\n\x06values\x18\x01 \x03(\x01\"\x18\n\x06\x41\x63tion\x12\x0e\n\x06\x61\x63tion\x18\x01 \x01(\x06\"E\n\nStepResult\x12\x15\n\x05state\x18\x01 \x01(\x0b\x32\x06.State\x12\x0e\n\x06reward\x18\x02 \x01(\x01\x12\x10\n\x08terminal\x18\x03 \x01(\x08\"\x07\n\x05\x45mpty2\x95\x01\n\x0b\x45nvironment\x12#\n\nInitialize\x12\x0b.Attributes\x1a\x06.Empty\"\x00\x12\x19\n\x05Start\x12\x06.Empty\x1a\x06.State\"\x00\x12\x1e\n\x04Step\x12\x07.Action\x1a\x0b.StepResult\"\x00\x12&\n\rGetAttributes\x12\x06.Empty\x1a\x0b.Attributes\"\x00\x32m\n\x05\x41gent\x12(\n\nInitialize\x12\x10.AgentAttributes\x1a\x06.Empty\"\x00\x12\x1a\n\x05Start\x12\x06.State\x1a\x07.Action\"\x00\x12\x1e\n\x04Step\x12\x0b.StepResult\x1a\x07.Action\"\x00\x42\nZ\x08.;remoteb\x06proto3'
)




_ATTRIBUTES = _descriptor.Descriptor(
  name='Attributes',
  full_name='Attributes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='attributes', full_name='Attributes.attributes', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16,
  serialized_end=48,
)


_AGENTATTRIBUTES = _descriptor.Descriptor(
  name='AgentAttributes',
  full_name='AgentAttributes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='experiment', full_name='AgentAttributes.experiment', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='environment', full_name='AgentAttributes.environment', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=134,
)


_STATE = _descriptor.Descriptor(
  name='State',
  full_name='State',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='values', full_name='State.values', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=159,
)


_ACTION = _descriptor.Descriptor(
  name='Action',
  full_name='Action',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='action', full_name='Action.action', index=0,
      number=1, type=6, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=161,
  serialized_end=185,
)


_STEPRESULT = _descriptor.Descriptor(
  name='StepResult',
  full_name='StepResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='StepResult.state', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reward', full_name='StepResult.reward', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='terminal', full_name='StepResult.terminal', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=187,
  serialized_end=256,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=258,
  serialized_end=265,
)

_AGENTATTRIBUTES.fields_by_name['experiment'].message_type = _ATTRIBUTES
_AGENTATTRIBUTES.fields_by_name['environment'].message_type = _ATTRIBUTES
_STEPRESULT.fields_by_name['state'].message_type = _STATE
DESCRIPTOR.message_types_by_name['Attributes'] = _ATTRIBUTES
DESCRIPTOR.message_types_by_name['AgentAttributes'] = _AGENTATTRIBUTES
DESCRIPTOR.message_types_by_name['State'] = _STATE
DESCRIPTOR.message_types_by_name['Action'] = _ACTION
DESCRIPTOR.message_types_by_name['StepResult'] = _STEPRESULT
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Attributes = _reflection.GeneratedProtocolMessageType('Attributes', (_message.Message,), {
  'DESCRIPTOR' : _ATTRIBUTES,
  '__module__' : 'remote_pb2'
  # @@protoc_insertion_point(class_scope:Attributes)
  })
_sym_db.RegisterMessage(Attributes)

AgentAttributes = _reflection.GeneratedProtocolMessageType('AgentAttributes', (_message.Message,), {
  'DESCRIPTOR' : _AGENTATTRIBUTES,
  '__module__' : 'remote_pb2'
  # @@protoc_insertion_point(class_scope:AgentAttributes)
  })
_sym_db.RegisterMessage(AgentAttributes)

State = _reflection.GeneratedProtocolMessageType('State', (_message.Message,), {
  'DESCRIPTOR' : _STATE,
  '__module__' : 'remote_pb2'
  # @@protoc_insertion_point(class_scope:State)
  })
_sym_db.RegisterMessage(State)

Action = _reflection.GeneratedProtocolMessageType('Action', (_message.Message,), {
  'DESCRIPTOR' : _ACTION,
  '__module__' : 'remote_pb2'
  # @@protoc_insertion_point(class_scope:Action)
  })
_sym_db.RegisterMessage(Action)

StepResult = _reflection.GeneratedProtocolMessageType('StepResult', (_message.Message,), {
  'DESCRIPTOR' : _STEPRESULT,
  '__module__' : 'remote_pb2'
  # @@protoc_insertion_point(class_scope:StepResult)
  })
_sym_db.RegisterMessage(StepResult)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'remote_pb2'
  # @@protoc_insertion_point(class_scope:Empty)
  })
_sym_db.RegisterMessage(Empty)


DESCRIPTOR._options = None

_ENVIRONMENT = _descriptor.ServiceDescriptor(
  name='Environment',
  full_name='Environment',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=268,
  serialized_end=417,
  methods=[
  _descriptor.MethodDescriptor(
    name='Initialize',
    full_name='Environment.Initialize',
    index=0,
    containing_service=None,
    input_type=_ATTRIBUTES,
    output_type=_EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Start',
    full_name='Environment.Start',
    index=1,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_STATE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Step',
    full_name='Environment.Step',
    index=2,
    containing_service=None,
    input_type=_ACTION,
    output_type=_STEPRESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetAttributes',
    full_name='Environment.GetAttributes',
    index=3,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_ATTRIBUTES,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_ENVIRONMENT)

DESCRIPTOR.services_by_name['Environment'] = _ENVIRONMENT


_AGENT = _descriptor.ServiceDescriptor(
  name='Agent',
  full_name='Agent',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  serialized_start=419,
  serialized_end=528,
  methods=[
  _descriptor.MethodDescriptor(
    name='Initialize',
    full_name='Agent.Initialize',
    index=0,
    containing_service=None,
    input_type=_AGENTATTRIBUTES,
    output_type=_EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Start',
    full_name='Agent.Start',
    index=1,
    containing_service=None,
    input_type=_STATE,
    output_type=_ACTION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Step',
    full_name='Agent.Step',
    index=2,
    containing_service=None,
    input_type=_STEPRESULT,
    output_type=_ACTION,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_AGENT)

DESCRIPTOR.services_by_name['Agent'] = _AGENT

# @@protoc_insertion_point(module_scope)
