# Taskboard evaluation fixture

Static Python fixture for parsing, resolution, and lexical retrieval. Run from this
directory if exercising its code directly. Connection and email delivery are
in-memory stand-ins. This is fixture data, not a deployed authentication service.

The call path is run_pipeline -> transform -> write_to_db -> Repository.save.
Transform normalizes the record before forwarding it to the persistence step.
