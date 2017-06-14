# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: spinn/util/logging.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='spinn/util/logging.proto',
  package='logging',
  syntax='proto2',
  serialized_pb=_b('\n\x18spinn/util/logging.proto\x12\x07logging\"V\n\x08SpinnLog\x12$\n\x06header\x18\x01 \x03(\x0b\x32\x14.logging.SpinnHeader\x12$\n\x07\x65ntries\x18\x02 \x03(\x0b\x32\x13.logging.SpinnEntry\"\x8c\x02\n\x0bSpinnHeader\x12\x14\n\x0ctotal_params\x18\x01 \x01(\x05\x12\x1a\n\x12model_architecture\x18\x02 \x01(\t\x12\x16\n\x0e\x65val_filenames\x18\x03 \x03(\t\x12\x12\n\nstart_step\x18\x04 \x01(\x05\x12\x12\n\nstart_time\x18\x05 \x01(\x03\x12\x13\n\x0bmodel_label\x18\x06 \x03(\t\x12\x33\n\x05\x66lags\x18\x64 \x03(\x0b\x32$.logging.SpinnHeader.CommandLineFlag\x12\x12\n\nextra_logs\x18\x65 \x03(\t\x1a-\n\x0f\x43ommandLineFlag\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"\xa1\x01\n\x08\x45valData\x12\x1b\n\x13\x65val_class_accuracy\x18\x02 \x01(\x02\x12 \n\x18\x65val_transition_accuracy\x18\x03 \x01(\x02\x12\x10\n\x08\x66ilename\x18\x04 \x01(\t\x12\x1e\n\x16time_per_token_seconds\x18\x05 \x01(\x02\x12\x13\n\x0breport_path\x18\x06 \x01(\t\x12\x0f\n\x07invalid\x18\x07 \x01(\x02\"\x87\x01\n\x0fRLSamplingStats\x12\r\n\x05t_idx\x18\x01 \x01(\x05\x12\x10\n\x08\x63rossing\x18\x02 \x01(\x02\x12\x0f\n\x07gold_lb\x18\x03 \x01(\t\x12\x0f\n\x07pred_tr\x18\x04 \x01(\t\x12\x0f\n\x07pred_ev\x18\x05 \x01(\t\x12\x0f\n\x07strg_tr\x18\x06 \x01(\t\x12\x0f\n\x07strg_ev\x18\x07 \x01(\t\"\xad\x04\n\nSpinnEntry\x12\x0c\n\x04step\x18\x01 \x01(\x05\x12\x16\n\x0e\x63lass_accuracy\x18\x02 \x01(\x02\x12\x1b\n\x13transition_accuracy\x18\x03 \x01(\x02\x12\x12\n\ntotal_cost\x18\x04 \x01(\x02\x12\x1a\n\x12\x63ross_entropy_cost\x18\x05 \x01(\x02\x12\x17\n\x0ftransition_cost\x18\x06 \x01(\x02\x12\x0f\n\x07l2_cost\x18\x07 \x01(\x02\x12\x1e\n\x16time_per_token_seconds\x18\x08 \x01(\x02\x12\x15\n\rlearning_rate\x18\t \x01(\x02\x12\x0f\n\x07invalid\x18\n \x01(\x02\x12\x13\n\x0bmodel_label\x18\x16 \x01(\t\x12\x13\n\x0bpolicy_cost\x18\x0b \x01(\x02\x12\x12\n\nvalue_cost\x18\x0c \x01(\x02\x12\x15\n\rmean_adv_mean\x18\r \x01(\x02\x12\x1f\n\x17mean_adv_mean_magnitude\x18\x0e \x01(\x02\x12\x14\n\x0cmean_adv_var\x18\x0f \x01(\x02\x12\x1e\n\x16mean_adv_var_magnitude\x18\x10 \x01(\x02\x12\x0f\n\x07\x65psilon\x18\x11 \x01(\x02\x12\x13\n\x0btemperature\x18\x12 \x01(\x02\x12%\n\nevaluation\x18\x13 \x03(\x0b\x32\x11.logging.EvalData\x12-\n\x0brl_sampling\x18\x14 \x03(\x0b\x32\x18.logging.RLSamplingStats\x12\x12\n\ncheckpoint\x18\x15 \x01(\t\"\x8c\x01\n\x0c\x45valSentence\x12\x13\n\x0bsentence_id\x18\x01 \x01(\x05\x12\x12\n\nprediction\x18\x02 \x01(\x05\x12\r\n\x05truth\x18\x03 \x01(\x05\x12\x0e\n\x06output\x18\x04 \x03(\x02\x12\x19\n\x11sent1_transitions\x18\x05 \x03(\x05\x12\x19\n\x11sent2_transitions\x18\x06 \x03(\x05\"5\n\tEvalBatch\x12(\n\tsentences\x18\x01 \x03(\x0b\x32\x15.logging.EvalSentence\"7\n\x10\x45valuationReport\x12#\n\x07\x62\x61tches\x18\x01 \x03(\x0b\x32\x12.logging.EvalBatch')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_SPINNLOG = _descriptor.Descriptor(
  name='SpinnLog',
  full_name='logging.SpinnLog',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='logging.SpinnLog.header', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='entries', full_name='logging.SpinnLog.entries', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=37,
  serialized_end=123,
)


_SPINNHEADER_COMMANDLINEFLAG = _descriptor.Descriptor(
  name='CommandLineFlag',
  full_name='logging.SpinnHeader.CommandLineFlag',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='logging.SpinnHeader.CommandLineFlag.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='logging.SpinnHeader.CommandLineFlag.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=349,
  serialized_end=394,
)

_SPINNHEADER = _descriptor.Descriptor(
  name='SpinnHeader',
  full_name='logging.SpinnHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='total_params', full_name='logging.SpinnHeader.total_params', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='model_architecture', full_name='logging.SpinnHeader.model_architecture', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_filenames', full_name='logging.SpinnHeader.eval_filenames', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='start_step', full_name='logging.SpinnHeader.start_step', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='start_time', full_name='logging.SpinnHeader.start_time', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='model_label', full_name='logging.SpinnHeader.model_label', index=5,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='flags', full_name='logging.SpinnHeader.flags', index=6,
      number=100, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='extra_logs', full_name='logging.SpinnHeader.extra_logs', index=7,
      number=101, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_SPINNHEADER_COMMANDLINEFLAG, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=126,
  serialized_end=394,
)


_EVALDATA = _descriptor.Descriptor(
  name='EvalData',
  full_name='logging.EvalData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='eval_class_accuracy', full_name='logging.EvalData.eval_class_accuracy', index=0,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_transition_accuracy', full_name='logging.EvalData.eval_transition_accuracy', index=1,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='filename', full_name='logging.EvalData.filename', index=2,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='time_per_token_seconds', full_name='logging.EvalData.time_per_token_seconds', index=3,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='report_path', full_name='logging.EvalData.report_path', index=4,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='invalid', full_name='logging.EvalData.invalid', index=5,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=397,
  serialized_end=558,
)


_RLSAMPLINGSTATS = _descriptor.Descriptor(
  name='RLSamplingStats',
  full_name='logging.RLSamplingStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='t_idx', full_name='logging.RLSamplingStats.t_idx', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='crossing', full_name='logging.RLSamplingStats.crossing', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gold_lb', full_name='logging.RLSamplingStats.gold_lb', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pred_tr', full_name='logging.RLSamplingStats.pred_tr', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pred_ev', full_name='logging.RLSamplingStats.pred_ev', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='strg_tr', full_name='logging.RLSamplingStats.strg_tr', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='strg_ev', full_name='logging.RLSamplingStats.strg_ev', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=561,
  serialized_end=696,
)


_SPINNENTRY = _descriptor.Descriptor(
  name='SpinnEntry',
  full_name='logging.SpinnEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='step', full_name='logging.SpinnEntry.step', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='class_accuracy', full_name='logging.SpinnEntry.class_accuracy', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='transition_accuracy', full_name='logging.SpinnEntry.transition_accuracy', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='total_cost', full_name='logging.SpinnEntry.total_cost', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cross_entropy_cost', full_name='logging.SpinnEntry.cross_entropy_cost', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='transition_cost', full_name='logging.SpinnEntry.transition_cost', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='l2_cost', full_name='logging.SpinnEntry.l2_cost', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='time_per_token_seconds', full_name='logging.SpinnEntry.time_per_token_seconds', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='logging.SpinnEntry.learning_rate', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='invalid', full_name='logging.SpinnEntry.invalid', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='model_label', full_name='logging.SpinnEntry.model_label', index=10,
      number=22, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='policy_cost', full_name='logging.SpinnEntry.policy_cost', index=11,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value_cost', full_name='logging.SpinnEntry.value_cost', index=12,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mean_adv_mean', full_name='logging.SpinnEntry.mean_adv_mean', index=13,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mean_adv_mean_magnitude', full_name='logging.SpinnEntry.mean_adv_mean_magnitude', index=14,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mean_adv_var', full_name='logging.SpinnEntry.mean_adv_var', index=15,
      number=15, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mean_adv_var_magnitude', full_name='logging.SpinnEntry.mean_adv_var_magnitude', index=16,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='epsilon', full_name='logging.SpinnEntry.epsilon', index=17,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='temperature', full_name='logging.SpinnEntry.temperature', index=18,
      number=18, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='evaluation', full_name='logging.SpinnEntry.evaluation', index=19,
      number=19, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rl_sampling', full_name='logging.SpinnEntry.rl_sampling', index=20,
      number=20, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='checkpoint', full_name='logging.SpinnEntry.checkpoint', index=21,
      number=21, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=699,
  serialized_end=1256,
)


_EVALSENTENCE = _descriptor.Descriptor(
  name='EvalSentence',
  full_name='logging.EvalSentence',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sentence_id', full_name='logging.EvalSentence.sentence_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='prediction', full_name='logging.EvalSentence.prediction', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='truth', full_name='logging.EvalSentence.truth', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output', full_name='logging.EvalSentence.output', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sent1_transitions', full_name='logging.EvalSentence.sent1_transitions', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sent2_transitions', full_name='logging.EvalSentence.sent2_transitions', index=5,
      number=6, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1259,
  serialized_end=1399,
)


_EVALBATCH = _descriptor.Descriptor(
  name='EvalBatch',
  full_name='logging.EvalBatch',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sentences', full_name='logging.EvalBatch.sentences', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1401,
  serialized_end=1454,
)


_EVALUATIONREPORT = _descriptor.Descriptor(
  name='EvaluationReport',
  full_name='logging.EvaluationReport',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batches', full_name='logging.EvaluationReport.batches', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1456,
  serialized_end=1511,
)

_SPINNLOG.fields_by_name['header'].message_type = _SPINNHEADER
_SPINNLOG.fields_by_name['entries'].message_type = _SPINNENTRY
_SPINNHEADER_COMMANDLINEFLAG.containing_type = _SPINNHEADER
_SPINNHEADER.fields_by_name['flags'].message_type = _SPINNHEADER_COMMANDLINEFLAG
_SPINNENTRY.fields_by_name['evaluation'].message_type = _EVALDATA
_SPINNENTRY.fields_by_name['rl_sampling'].message_type = _RLSAMPLINGSTATS
_EVALBATCH.fields_by_name['sentences'].message_type = _EVALSENTENCE
_EVALUATIONREPORT.fields_by_name['batches'].message_type = _EVALBATCH
DESCRIPTOR.message_types_by_name['SpinnLog'] = _SPINNLOG
DESCRIPTOR.message_types_by_name['SpinnHeader'] = _SPINNHEADER
DESCRIPTOR.message_types_by_name['EvalData'] = _EVALDATA
DESCRIPTOR.message_types_by_name['RLSamplingStats'] = _RLSAMPLINGSTATS
DESCRIPTOR.message_types_by_name['SpinnEntry'] = _SPINNENTRY
DESCRIPTOR.message_types_by_name['EvalSentence'] = _EVALSENTENCE
DESCRIPTOR.message_types_by_name['EvalBatch'] = _EVALBATCH
DESCRIPTOR.message_types_by_name['EvaluationReport'] = _EVALUATIONREPORT

SpinnLog = _reflection.GeneratedProtocolMessageType('SpinnLog', (_message.Message,), dict(
  DESCRIPTOR = _SPINNLOG,
  __module__ = 'spinn.util.logging_pb2'
  # @@protoc_insertion_point(class_scope:logging.SpinnLog)
  ))
_sym_db.RegisterMessage(SpinnLog)

SpinnHeader = _reflection.GeneratedProtocolMessageType('SpinnHeader', (_message.Message,), dict(

  CommandLineFlag = _reflection.GeneratedProtocolMessageType('CommandLineFlag', (_message.Message,), dict(
    DESCRIPTOR = _SPINNHEADER_COMMANDLINEFLAG,
    __module__ = 'spinn.util.logging_pb2'
    # @@protoc_insertion_point(class_scope:logging.SpinnHeader.CommandLineFlag)
    ))
  ,
  DESCRIPTOR = _SPINNHEADER,
  __module__ = 'spinn.util.logging_pb2'
  # @@protoc_insertion_point(class_scope:logging.SpinnHeader)
  ))
_sym_db.RegisterMessage(SpinnHeader)
_sym_db.RegisterMessage(SpinnHeader.CommandLineFlag)

EvalData = _reflection.GeneratedProtocolMessageType('EvalData', (_message.Message,), dict(
  DESCRIPTOR = _EVALDATA,
  __module__ = 'spinn.util.logging_pb2'
  # @@protoc_insertion_point(class_scope:logging.EvalData)
  ))
_sym_db.RegisterMessage(EvalData)

RLSamplingStats = _reflection.GeneratedProtocolMessageType('RLSamplingStats', (_message.Message,), dict(
  DESCRIPTOR = _RLSAMPLINGSTATS,
  __module__ = 'spinn.util.logging_pb2'
  # @@protoc_insertion_point(class_scope:logging.RLSamplingStats)
  ))
_sym_db.RegisterMessage(RLSamplingStats)

SpinnEntry = _reflection.GeneratedProtocolMessageType('SpinnEntry', (_message.Message,), dict(
  DESCRIPTOR = _SPINNENTRY,
  __module__ = 'spinn.util.logging_pb2'
  # @@protoc_insertion_point(class_scope:logging.SpinnEntry)
  ))
_sym_db.RegisterMessage(SpinnEntry)

EvalSentence = _reflection.GeneratedProtocolMessageType('EvalSentence', (_message.Message,), dict(
  DESCRIPTOR = _EVALSENTENCE,
  __module__ = 'spinn.util.logging_pb2'
  # @@protoc_insertion_point(class_scope:logging.EvalSentence)
  ))
_sym_db.RegisterMessage(EvalSentence)

EvalBatch = _reflection.GeneratedProtocolMessageType('EvalBatch', (_message.Message,), dict(
  DESCRIPTOR = _EVALBATCH,
  __module__ = 'spinn.util.logging_pb2'
  # @@protoc_insertion_point(class_scope:logging.EvalBatch)
  ))
_sym_db.RegisterMessage(EvalBatch)

EvaluationReport = _reflection.GeneratedProtocolMessageType('EvaluationReport', (_message.Message,), dict(
  DESCRIPTOR = _EVALUATIONREPORT,
  __module__ = 'spinn.util.logging_pb2'
  # @@protoc_insertion_point(class_scope:logging.EvaluationReport)
  ))
_sym_db.RegisterMessage(EvaluationReport)


# @@protoc_insertion_point(module_scope)
