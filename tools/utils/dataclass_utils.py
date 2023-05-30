import dataclasses


def get_field(d: dataclasses.dataclass, key: str) -> dataclasses.Field:
    """
    Given a dataclass and a field name, return the field object.

    Useful when trying to grab metadata about a field from a dataclass.

    Ex.
    ```
    @dataclass
    class D:
        a: int = 1

    print(dataclass_utils.get_field(D, "a").default) # Prints 1
    ```
    """
    fields = dataclasses.fields(d)
    for field in fields:
        if field.name == key:
            return field
    raise ValueError(f"Could not find field {key} in {d}")
