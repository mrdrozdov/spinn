import datetime
import sys
import json


class Logger(object):
    # A logging alternative that doesn't leave logs open between writes,
    # so as to allow AFS synchronization.

    # Level constants
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self, log_path=None, json_log_path=None,
                 show_level=False, min_print_level=0, min_file_level=0):
        # log_path: The full path for the log file to write. The file will be appended
        #   to if it exists.
        # min_print_level: Only messages with level above this level will be printed to stderr.
        # min_file_level: Only messages with level above this level will be
        #   written to disk.
        self.log_path = log_path
        self.json_log_path = json_log_path
        self.min_print_level = min_print_level
        self.min_file_level = min_file_level
        self.show_level = show_level

    def Log(self, message, level=INFO):
        if level >= self.min_print_level:
            # Write to STDERR
            sys.stderr.write("[%i] %s\n" % (level, message))
        if self.log_path and level >= self.min_file_level:
            # Write to the log file then close it
            with open(self.log_path, 'a') as f:
                datetime_string = datetime.datetime.now().strftime(
                    "%y-%m-%d %H:%M:%S")
                if self.show_level:
                    f.write("%s [%i] %s\n" % (datetime_string, level, message))
                else:
                    f.write("%s %s\n" % (datetime_string, message))

    def LogJSON(self, message_obj, level=INFO):
        if self.json_log_path and level >= self.min_file_level:
            with open(self.json_log_path, 'w') as f:
                print >>f, json.dumps(message_obj)
        else:
            sys.stderr.write('WARNING: No JSON log filename.')


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
            print_formatter=None):
        # root: The parent log message to store in file (SpinnLog).
        #   The message must contain only "header" and "entries" as submessages.
        #   Fill out the header when creating the logger.
        # log_path: The full path for the log file to write. The file will be
        #   appended to if it exists.
        # min_print_level: Only messages with level above this level will be
        #   printed to stderr.
        # print_formatter: Function to format the stderr-printed message.
        #   If not set, prints the entire textproto.

        super(ProtoLogger, self).__init__()
        self.root = None
        self.log_path = log_path
        self.json_log_path = json_log_path
        self.min_print_level = min_print_level
        self.print_formatter = print_formatter
        if self.print_formatter is None:
            self.print_formatter = default_formatter

    def LogHeader(self, header):
        if self.root is not None:
            raise Exception('Root object already logged!')
        self.root = pb.SpinnLog()
        self.root.header.MergeFrom(header)

        # Store the header.
        if self.log_path:  # Write to the log file then close it
            # Truncate the log for the first run.
            with open(self.log_path, 'w') as f:
                f.write(str(self.root))
        self.root.Clear()

    def Log(self, message, level=INFO):
        if level >= self.min_print_level:
            # Write to STDERR
            sys.stderr.write("[%i] %s\n" % (level, message))

    def LogEntry(self, message, level=INFO):
        if self.root is None:
            raise Exception('Log the header first!')
        self.root.entries.add().MergeFrom(message)
        try:
            msg_str = str(self.root)
            if level >= self.min_print_level:  # Write to stderr
                msg = self.print_formatter(message)
                sys.stderr.write("[%i] %s\n" % (level, msg))
            if self.log_path:  # Write to the log file then close it
                with open(self.log_path, 'a') as f:
                    f.write(msg_str)
        finally:
            self.root.Clear()
