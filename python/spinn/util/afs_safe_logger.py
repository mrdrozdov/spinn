import datetime
import sys
import json
import logging_pb2 as pb
import re


def default_formatter(log_entry):
    fmt = 'step {}: class_acc{}, transition_acc{}, total_cost{}'.format(
        log_entry.step,
        log_entry.class_accuracy,
        log_entry.transition_accuracy,
        log_entry.total_cost)
    return fmt


class ProtoLogger(object):
    '''Writes logs in textproto format, so it is both human and machine
    readable. Writing text is not as efficient as binary, but the advantage is
    we can somewhat extend the message by appending the file.'''

    # Level constants
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(
            self,
            log_path=None,
            json_log_path=None,
            min_print_level=1,
            print_formatter=None,
            write_proto=True):
        # root: The parent log message to store in file (SpinnLog).
        #   The message must contain only "header" and "entries" as submessages.
        #   Fill out the header when creating the logger.
        # log_path: The full path for the log file to write. The file will be
        #   appended to if it exists.
        # min_print_level: Only messages with level above this level will be
        #   printed to stderr.
        # print_formatter: Function to format the stderr-printed message.
        #   If not set, prints the entire textproto.
        # write_proto: (default True) Write proto format to file. Otherwise,
        #   write what was printed to file.

        super(ProtoLogger, self).__init__()
        self.root = None
        self.log_path = log_path
        self.json_log_path = json_log_path
        self.min_print_level = min_print_level
        self.print_formatter = print_formatter
        self.write_proto = write_proto
        if self.print_formatter is None:
            self.print_formatter = default_formatter
        with open(self.log_path, 'w') as f:
            # Truncate the file.
            pass

    def LogHeader(self, header):
        if self.root is not None:
            raise Exception('Root object already logged!')
        self.root = pb.SpinnLog()
        self.root.header.MergeFrom(header)

        # Store the header.
        if self.log_path and self.write_proto:
            # Truncate the log for the first run.
            with open(self.log_path, 'a') as f:
                f.write(str(self.root))
        self.root.Clear()

    def Log(self, message, level=INFO):
        if level < self.min_print_level:
            return
        # Write to STDERR
        msg_str = "[%i] %s\n" % (level, message)
        sys.stderr.write(msg_str)
        if self.log_path and not self.write_proto:
            with open(self.log_path, 'a') as f:
                f.write(msg_str)

    def LogEntry(self, message, level=INFO):
        if self.root is None:
            raise Exception('Log the header first!')
        self.root.entries.add().MergeFrom(message)
        try:
            msg_str = str(self.root)
            msg_fmt =  self.print_formatter(message)
            datetime_string = datetime.datetime.now().strftime(
                "%y-%m-%d %H:%M:%S ")
            msg_line = re.sub('^', datetime_string, msg_fmt, flags=re.MULTILINE) + '\n'
            if level >= self.min_print_level:  # Write to stderr
                sys.stderr.write(msg_line)
            if not self.write_proto:
                msg_str = msg_line
            if self.log_path:  # Write to the log file then close it
                with open(self.log_path, 'a') as f:
                    f.write(msg_str)
        finally:
            self.root.Clear()
