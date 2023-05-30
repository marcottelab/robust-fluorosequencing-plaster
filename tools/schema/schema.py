import base64
import logging

from munch import Munch

log = logging.getLogger(__name__)


class SchemaInvalid(Exception):
    """
    Raised when the schema itself is invalid.
    """

    def __init__(self, help):
        self.help = help


class SchemaValidationFailed(Exception):
    """
    Raised when a test fails to conform to the given schema.
    """

    def __init__(self, help=None, context=None, errors=None):
        self.help = help
        self.context = context
        self.errors = errors

    def __str__(self):
        return f"SchemaValidationFailed {self.help} {self.context if self.context is not None else ''} {self.errors if self.errors is not None else ''}"


def schema_check(expr, help=None):
    if not expr:
        raise SchemaValidationFailed(help=help)


TUP_VALIDATOR = 0
TUP_KWS = 1
TUP_ELEMS = 2
TUP_TYPE = 3


class Schema:
    """
    A simple schema validator.

    Example #1: A single-level dict

        import plaster.tools.schema.schema.Schema as s
        schema = s(
            s.is_kws(
                a=s.is_int(),
                b=s.is_int(),
            )
        )
        test = dict(a=1, b="not an int")
        schema.validate(test)

    Example #2: A multi-level structure

        schema = s(
            s.is_dict(required=True, no_extras=True, ignore_underscored_keys=True, elems=dict(
                an_optional_int=s.is_int(),
                some_required_int=s.is_int(required=True),
                some_other_required_int_that_can_be_none=s.is_int(required=True, noneable=True),
                a_key_with_a_bound=s.is_int(min_val=0, max_val=5),
                a_simple_optional_list=s.is_list(s.is_int()),
                a_bounded_list_of_ints=s.is_list(min_len=0, max_len=3, elems=s.is_int()),
                a_deprecated_field=s.is_deprecated(),
                a_bool=s.is_bool(),
                an_sub_dict_with_minimal_rules=s.is_kws(
                    a=s.is_int(),
                    b=s.is_int(),
                ),
            )

    Example #3: Including help

        schema = s(
            s.is_dict(help="Important things", elems=dict(
                peace=s.is_int(help="Peace man!"),
                and=s.is_int(),
                love=s.is_int(help="is all it takes."),
            )
        schema.help()


    Usage if you want to validate the options of a function:

        def some_func(n_pres, dyes, **kws):
            schema.validate(dict(locals(), **kws))

    Validators:
        Options applicable to all validators:
            required=False
            noneable=False
            help="A string that documents this field"

        is_int(bounds=None)
        is_float(bounds=None)
        is_number(bounds=None)
        is_str()

        is_list(min_len=None, max_len=None, elems=sub_element_schema)
        is_dict(
            ignore_underscored_keys=False,
            all_required=False,
            no_extras=False,
            use_default_on_none=False,
            elems=sub_element_schema
        )
        is_kws(sub_element_schema)  # Like dict but uses **kws for sub_elems
        is_kws_r(sub_element_schema)  # Like dict but uses **kws for sub_elems and asserts all required

    TASK: Document defaults
    """

    def __init__(self, schema_tuple):
        Schema._check_is_schema_tuple(schema_tuple)
        self.schema_tuple = schema_tuple

    def schema(self):
        """
        Example:
            schema0 = s(
                s.is_kws_r(
                    var1=s.is_int()
                )
            )

            schema1 = s(
                s.is_kws_r(
                    **schema0.schema(),
                    var2=s.is_str(),
                )
            )
        """
        return self.schema_tuple[TUP_ELEMS]

    @staticmethod
    def _print_error(message):
        """mock-point"""
        log.error(message)

    @staticmethod
    def _print_help(indent, key, help=None):
        """mock-point"""
        yellow = "\u001b[33m"
        reset = "\u001b[0m"

        print(
            f"{reset}{indent}{key}: "
            f"{yellow if help else ''}"
            f"{help if help is not None else 'No help available'}{reset}"
        )

    def help(self):
        def _recurse_help(schema_tuple, level, context):
            Schema._print_help(
                " " * (level * 2), context, schema_tuple[TUP_KWS].get("help")
            )
            if schema_tuple[TUP_ELEMS] is not None:
                for key, elem_schema_tuple in schema_tuple[TUP_ELEMS].items():
                    _recurse_help(elem_schema_tuple, level + 1, key)

        _recurse_help(self.schema_tuple, 0, "root")

    def validate(
        self, to_test, print_on_error=True, raise_on_error=True, context="root"
    ):
        error = self._recurse(self.schema_tuple, to_test, context)

        def _recurse_print(e, level):
            if e.context is not None:
                Schema._print_error(f"{' ' * (level * 2)}In context of {e.context}:")
                level += 1

            if e.errors is not None:
                [_recurse_print(_e, level) for _e in e.errors]
            else:
                Schema._print_error(f"{' ' * (level * 2)}{e.help}")

        if error is not None:
            if print_on_error:
                _recurse_print(error, 0)
            if raise_on_error:
                raise error
            return False
        return True

    def apply_defaults(self, defaults, apply_to, override_nones=False):
        """Apply defaults to to_apply dict (and sub-dicts)."""

        def _recurse(schema_tuple, _defaults, _apply_to):
            if schema_tuple[TUP_ELEMS] is not None and schema_tuple[TUP_TYPE] is dict:
                assert isinstance(_defaults, (dict, type(None)))
                assert isinstance(_apply_to, dict)
                elems = schema_tuple[TUP_ELEMS]
                # if elems is None:
                #     elems = {}
                assert isinstance(elems, dict)

                # APPLY default to anything that is in the defaults that *is* in the
                # elems schema and that isn't in apply_to already (or is none if override_nones)
                # NOTE: treats empty lists as equivalent to None if override_nones, but then
                # we must ensure that def_val is not None since schema may require a list.
                for def_key, def_val in _defaults.items():
                    if def_key in elems and (
                        def_key not in _apply_to
                        or (
                            def_val is not None
                            and def_key in _apply_to
                            and _apply_to[def_key] in [None, []]
                            and override_nones
                        )
                    ):
                        _apply_to[def_key] = def_val

                for key, elem_schema_tuple in elems.items():
                    if (
                        key in _defaults
                        and _defaults[key] is not None
                        and key in _apply_to
                        and _apply_to[key] is not None
                    ):
                        # _defaults[key] can be None in the situation where there is
                        # a perfectly good dict in a sub key of the apply_to already.
                        _recurse(elem_schema_tuple, _defaults[key], _apply_to[key])

        _recurse(self.schema_tuple, defaults, apply_to)

    def top_level_fields(self):
        """
        Return all *top-level* fields (Does NOT recurse).
        Returns a list of tuples
            [
                (field_name, field_type, field_help, field_userdata, field_subtype)
            ]

            field_type is converted into a Python type (list, dict, int, float, str)
            field_help returns None if help is not given.
            field_subtype is only used for lists. It is the type of the list element.
        """
        validator_fn, kws, sub_elems, type_ = self.schema_tuple
        if sub_elems is None:
            return []

        fields = []
        for name, obj in sub_elems.items():
            field_subtype = None
            if obj[TUP_TYPE] is list:
                field_subtype = obj[TUP_ELEMS][TUP_TYPE]

            fields += [
                (
                    name,
                    obj[TUP_TYPE],
                    obj[TUP_KWS].get("help"),
                    obj[TUP_KWS].get("userdata"),
                    field_subtype,
                )
            ]
        return fields

    def requirements(self):
        """
        Return all *top-level* required fields (Does NOT recurse).
        Returns a list of tuples
            [
                (field_name, field_type, field_help, field_userdata)
            ]

            field_type is converted into a Python type (list, dict, int, float, str)
            field_help returns None if help is not given.
        """
        validator_fn, kws, sub_elems, type_ = self.schema_tuple
        if sub_elems is None:
            return []

        all_req = kws.get("all_required", False)
        required = []
        for name, obj in sub_elems.items():
            if all_req or (
                obj[TUP_ELEMS] is not None and obj[TUP_ELEMS].get("required")
            ):
                required += [
                    (
                        name,
                        obj[TUP_TYPE],
                        obj[TUP_KWS].get("help"),
                        obj[TUP_KWS].get("userdata"),
                    )
                ]
        return required

    @classmethod
    def _recurse(cls, schema_tuple, to_test, context):
        """Recurse check sub_schema"""
        try:
            schema_tuple[TUP_VALIDATOR](to_test)
        except SchemaValidationFailed as e:
            e.context = context
            return e
        return None

    @classmethod
    def _check(cls, expr, help):
        return schema_check(expr, help)

    @classmethod
    def _check_is_type(cls, arg, types):
        """Leaf-node type check"""
        cls._check(
            isinstance(arg, types),
            f"Must be of type '{','.join([t.__name__ for t in types])}' (was {type(arg).__name__}).",
        )

    @classmethod
    def _check_errors(cls, errors):
        """
        Used by contexts after they have accumulated their list of errors;
        raise if there is any error in the list.
        """
        errors = [error for error in errors if error is not None]
        if len(errors) > 0:
            raise SchemaValidationFailed(errors=errors)

    @classmethod
    def _check_noneable(cls, kws, arg):
        """Check noneable flag and return True if None and allowed"""
        if not kws.get("noneable") and arg is None:
            raise SchemaValidationFailed(help=f"Was None but None is not allowed.")
        return arg is None

    @classmethod
    def _check_is_schema_tuple(cls, schema_tuple):
        """
        Check that the validator itself returns a properly constructed schema tuple
        (callable, type name, sub_elems)
        """
        if schema_tuple is None:
            return

        if not isinstance(schema_tuple, tuple) or len(schema_tuple) != 4:
            raise SchemaInvalid("a Schema requires a 4 tuple")

        if not isinstance(schema_tuple[TUP_TYPE], (type, type(None))):
            raise SchemaInvalid("a Schema requires returning a type")

        if not isinstance(schema_tuple, tuple):
            raise SchemaInvalid("a Schema was required")

        if not callable(schema_tuple[TUP_VALIDATOR]):
            raise SchemaInvalid(
                "a Schema was required (tuple[TUP_VALIDATOR] not callable)"
            )

        if not isinstance(schema_tuple[TUP_KWS], dict):
            raise SchemaInvalid("a Schema was required (tuple[TUP_KWS] not a dict)")

    @classmethod
    def _check_arg_type(cls, arg, arg_name, expected_types):
        """
        Check that the validator argument is of the right type.
        """
        if not isinstance(arg, expected_types):
            raise SchemaInvalid(
                f"a Schema expected argument '{arg_name}' to be of type(s) '{expected_types}' "
                f"(was '{type(arg).__name__}'.)"
            )

    @classmethod
    def _check_bounds_arg(cls, bounds):
        """Helper to check validity of the bounds argument"""
        if bounds is not None:
            cls._check_arg_type(bounds, "bounds", tuple)
            if len(bounds) != 2:
                raise SchemaInvalid(
                    f"bounds parameter should be length 2. (was {len(bounds)})"
                )

            cls._check_arg_type(bounds[0], "bounds[0]", (int, float, type(None)))
            cls._check_arg_type(bounds[1], "bounds[1]", (int, float, type(None)))

    @classmethod
    def _check_bounds(cls, arg, bounds=None):
        """Helper to check bounds for int, float, number"""
        if bounds is not None:
            if bounds[0] is not None:
                cls._check(arg >= bounds[0], f"Must be >= {bounds[0]} (was {arg}).")
            if bounds[1] is not None:
                cls._check(arg <= bounds[1], f"Must be <= {bounds[1]} (was {arg}).")

    @classmethod
    def is_int(cls, bounds=None, **kws):
        cls._check_bounds_arg(bounds)

        def validator(arg):
            if cls._check_noneable(kws, arg):
                return
            cls._check_is_type(arg, (int,))
            cls._check_bounds(arg, bounds)

        return validator, kws, None, int

    @classmethod
    def is_float(cls, bounds=None, **kws):
        cls._check_bounds_arg(bounds)

        def validator(arg):
            if cls._check_noneable(kws, arg):
                return
            cls._check_is_type(arg, (float,))
            cls._check_bounds(arg, bounds)

        return validator, kws, None, float

    @classmethod
    def is_number(cls, bounds=None, **kws):
        cls._check_bounds_arg(bounds)

        def validator(arg):
            if cls._check_noneable(kws, arg):
                return
            cls._check_is_type(arg, (int, float))
            cls._check_bounds(arg, bounds)

        return validator, kws, None, float

    @classmethod
    def is_str(cls, **kws):
        allow_empty_string = kws.get("allow_empty_string", True)
        options = kws.get("options", None)

        def validator(arg):
            if cls._check_noneable(kws, arg):
                return
            cls._check_is_type(arg, (str,))
            if not allow_empty_string:
                cls._check(arg != "", f"Empty string not allowed.")
            if options is not None:
                assert isinstance(options, (list, tuple))
                cls._check(arg in options, f"String '{arg}' not in allowed options.")

        return validator, kws, None, str

    @classmethod
    def is_bool(cls, **kws):
        def validator(arg):
            if cls._check_noneable(kws, arg):
                return
            cls._check_is_type(arg, (bool,))

        return validator, kws, None, bool

    @classmethod
    def is_list(cls, elems=None, **kws):
        min_len = kws.get("min_len")
        max_len = kws.get("max_len")

        cls._check_arg_type(min_len, "min_len", (int, type(None)))
        cls._check_arg_type(max_len, "max_len", (int, type(None)))
        cls._check_is_schema_tuple(elems)

        def validator(arg):
            if cls._check_noneable(kws, arg):
                return
            cls._check_is_type(arg, (list,))
            len_ = len(arg)
            if min_len is not None:
                cls._check(
                    len_ >= min_len,
                    f"Must be a list with at least {min_len} elements (had {len_}).",
                )
            if max_len is not None:
                cls._check(
                    len_ <= max_len,
                    f"Must be a list with at most {max_len} elements (had {len_}).",
                )

            if elems is not None:
                errors = []
                for i, elem in enumerate(arg):
                    errors += [cls._recurse(elems, elem, context=f"list element [{i}]")]
                cls._check_errors(errors)

        return validator, kws, elems, list

    @classmethod
    def is_dict(cls, elems=None, **kws):
        ignore_underscored_keys = kws.get("ignore_underscored_keys", False)
        all_required = kws.get("all_required", False)
        no_extras = kws.get("no_extras", False)

        if not isinstance(elems, (dict, type(None))):
            raise SchemaInvalid(
                f"An is_dict schema must have elems= of type dict. (was {type(elems).__name__})"
            )

        def validator(arg):
            if cls._check_noneable(kws, arg):
                return

            cls._check_is_type(arg, (dict,))

            errors = []
            if elems is not None:
                for key, elem_schema_tuple in elems.items():
                    if elem_schema_tuple is None:
                        continue

                    cls._check(
                        isinstance(key, str), f"The key '{key}' is not itself a string."
                    )

                    if ignore_underscored_keys and key[0] == "_":
                        continue

                    cls._check_is_schema_tuple(elem_schema_tuple)

                    if key not in arg:
                        if all_required or elem_schema_tuple[TUP_KWS].get("required"):
                            # Unless specifically overloaded...
                            if (
                                elem_schema_tuple[TUP_KWS].get("required", None)
                                is not False
                            ):
                                errors += [
                                    SchemaValidationFailed(
                                        help=f"Dict key '{key}' was required but not found."
                                    )
                                ]

                    if key in arg:
                        errors += [
                            cls._recurse(
                                elem_schema_tuple, arg[key], context=f"dict key '{key}'"
                            )
                        ]

            if no_extras:
                for key in arg.keys():
                    if ignore_underscored_keys and key[0] == "_":
                        continue

                    if key not in elems:
                        errors += [
                            SchemaValidationFailed(
                                help=f"Dict key '{key}' was present but not allowed."
                            )
                        ]

            cls._check_errors(errors)

        return validator, kws, elems, dict

    @classmethod
    def is_type(cls, type_, **kws):
        def validator(arg):
            if cls._check_noneable(kws, arg):
                return
            cls._check_is_type(arg, (type_,))

        return validator, kws, None, type_

    @classmethod
    def is_kws(cls, **kws):
        # A shortcut for a default is_dict
        return cls.is_dict(elems=kws)

    @classmethod
    def is_kws_r(cls, **kws):
        # A shortcut for a is_dict with all keys required
        return cls.is_dict(all_required=True, elems=kws)

    @classmethod
    def is_deprecated(cls, **kws):
        def validator(_):
            raise SchemaValidationFailed(help="Is deprecated.")

        kws["required"] = False
        return validator, kws, None, None

    @classmethod
    def is_base64(cls, **kws):
        options = kws.get("options", None)

        def validator(arg):
            if cls._check_noneable(kws, arg):
                return
            cls._check_is_type(arg, (str,))
            if options is not None:
                assert isinstance(options, (list, tuple))
                cls._check(arg in options, f"String '{arg}' not in allowed options.")
            try:
                base64.decode(arg)
                return
            except Exception:
                raise SchemaValidationFailed(help="Was not base64 decodable")

        return validator, kws, None, str


class Params(Munch):
    """
    A base class for a set of parameters that are validated with a schema
    """

    # Overload these in the subclass.
    defaults = None
    schema = None

    def _validate(self, cond, message):
        """
        A helper to assert a condition with a message
        """
        if not cond:
            raise SchemaValidationFailed(help=message)

    def validate(self):
        self.schema.apply_defaults(self.defaults, apply_to=self)
        self.schema.validate(self, context=self.__class__.__name__)

    def __init__(self, **kwargs):
        super().__init__(self.defaults)
        self.update(**kwargs)
        self.validate()
