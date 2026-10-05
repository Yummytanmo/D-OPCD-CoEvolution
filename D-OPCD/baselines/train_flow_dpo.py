#!/usr/bin/env python
try:
    from .training import main, parse_args
except ImportError:
    from training import main, parse_args


if __name__ == "__main__":
    main(parse_args("flow_dpo"))
