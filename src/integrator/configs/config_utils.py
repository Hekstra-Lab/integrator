from dataclasses import fields


def shallow_dict(dc) -> dict:
    return {f.name: getattr(dc, f.name) for f in fields(dc)}
