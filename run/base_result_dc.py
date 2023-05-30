import dataclasses
import logging
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

import flytekit
import numpy as np
import pandas as pd
import pyarrow
import yaml
from dataclasses_json import DataClassJsonMixin
from flytekit.types.directory import FlyteDirectory

from plaster.run import priors
from plaster.run.base_result import ArrayResult
from plaster.run.rf_train_v2.sklearn_rf import SKLearnRandomForestClassifier
from plaster.run.sigproc_v2.sigproc_v2_null_hypothesis import NullHypothesisPrecompute
from plaster.tools.classifier_helpers.call_bag import CallBag

log = logging.getLogger(__name__)


@dataclass
class LazyField(DataClassJsonMixin):
    """
    LazyField provides a mechanism for saving and loading dataclass fields to disk in an explicit way.

    LazyField can only be used as fields for dataclasses which subclass BaseResultDC.

    LazyField is implemented as a dataclass such that it can serialize the metadata needed to load the actual data from disk.

    Different types of data have different methods for serialization. To add support for new datatypes,
    add cases to the conditional in the save and get methods.
    """

    path: Optional[str] = None

    NPY_EXTENSION = ".npy"
    PICKLE_EXTENSION = ".pkl"

    def __post_init__(self):
        """
        The variables initialized here are necessary for the functionality of this field,
        but are not intended to be serialized.
        """

        # The name of the field in the dataclass that points to this LazyField
        self.field_name: str = None
        # This is the cached value
        self.value: Any = None
        # This is the folder for the current context
        self.folder: Path = None
        # The type of the field
        self.type: Type = None

    @staticmethod
    def _save_pkl(
        folder: Path,
        field_name: str,
        value: object,
    ) -> str:
        """Dump obj to path as a pickled string.

        Arguments:
            path: the destination file path without suffix
            obj: Python object to dump

        Returns:
            Relative (to result folder) path of pickle file
        """
        relpath = field_name + LazyField.PICKLE_EXTENSION
        full_path = folder / relpath
        full_path.write_bytes(pickle.dumps(value))
        return relpath

    @staticmethod
    def _save_np(folder: Path, field_name: str, value: np.ndarray) -> str:
        """Save numpy ndarray to path.

        Arguments:
            path: the destination file path without suffix
            arr: numpy ndarray to save

        Returns:
            Relative (to result folder) path of array file
        """
        relpath = field_name + LazyField.NPY_EXTENSION
        full_path = folder / relpath
        np.save(full_path, value)
        return relpath

    @staticmethod
    def _save_pd(folder: Path, field_name: str, value: pd.DataFrame) -> str:
        """Write pandas dataframe to given file path.

        Arguments:
            path: the destination file path
            df: pandas dataframe to save

        Returns:
            Relative (to result folder) path of data frame file
        """
        relpath = field_name + LazyField.PICKLE_EXTENSION
        full_path = folder / relpath

        batch = pyarrow.record_batch(value)

        with open(full_path, "wb") as f:
            with pyarrow.ipc.new_stream(f, batch.schema) as writer:
                writer.write_batch(batch)

        return relpath

    def set(self, value):
        self.value = value

    def save(self):
        if self.value is None:
            return

        save_handlers = {
            np.ndarray: LazyField._save_np,
            priors.Priors: LazyField._save_pkl,
            ArrayResult: LazyField._save_pkl,
            SKLearnRandomForestClassifier: LazyField._save_pkl,
            pd.DataFrame: LazyField._save_pd,
            CallBag: LazyField._save_pkl,
            NullHypothesisPrecompute: LazyField._save_pkl,
        }

        try:
            self.path = save_handlers[self.type](
                folder=self.folder, field_name=self.field_name, value=self.value
            )
        except KeyError:
            raise Exception(f"Unknown type {self.type}")

    def get(self):
        if self.value is None:
            if self.path is not None:
                if self.type == np.ndarray:
                    self.value = np.load(self.folder / self.path)
                elif self.type == priors.Priors:
                    self.value = pickle.loads((self.folder / self.path).read_bytes())
                elif self.type == ArrayResult:
                    self.value = pickle.loads((self.folder / self.path).read_bytes())
                    # change arr state filename to point at the local arr if it exists
                    state = self.value.__getstate__()
                    local_arr_path = self.folder / Path(state[0]).name
                    if local_arr_path.exists() and state[0] != local_arr_path:
                        self.value.__setstate__((str(local_arr_path), *state[1:]))
                elif self.type == SKLearnRandomForestClassifier:
                    self.value = pickle.loads((self.folder / self.path).read_bytes())
                elif self.type == CallBag:
                    self.value = pickle.loads((self.folder / self.path).read_bytes())
                elif self.type == NullHypothesisPrecompute:
                    self.value = pickle.loads((self.folder / self.path).read_bytes())
                elif self.type == pd.DataFrame:
                    with open(self.folder / self.path, "rb") as f:
                        with pyarrow.ipc.open_stream(f) as reader:
                            self.value = reader.read_next_batch().to_pandas()

                else:
                    raise Exception(f"Unknown type {self.type}")

        return self.value


@dataclass
class BaseResultDC:
    """
    Dataclasses that subclass BaseResultDC will behave largely the same as vanilla dataclasses,
    except for LazyField fields, which will serialize their value to disk on save, and won't load
    when the subclassed dataclass is deserialized until explcitly asked for.

    Example usage:

    @dataclass
    class MyResultDC(BaseResultDC):
        arr: LazyField = LazyField(np.ndarray)

    result = MyResultDC()
    result.arr.set(np.array([1, 2, 3]))
    result.serialize()

    folder = Path(tmpdir)
    serialized_json = result_a.serialize(folder)

    loaded_result = MyResult.deserialize(serialized_json, folder)
    print(loaded_result.arr.get()) # prints [1, 2, 3]
    """

    def __post_init__(self):
        if not hasattr(self, "_folder"):
            self._folder = None

        for f in dataclasses.fields(self):
            if f.type == LazyField:
                # If a value was provided in the constructor, then we need to replace the value with
                # A LazyField object containing that value
                value = getattr(self, f.name)
                if not isinstance(value, LazyField):
                    lazy_field = LazyField()
                    lazy_field.set(value)

                    if self._folder:
                        lazy_field.folder = self._folder

                    setattr(self, f.name, lazy_field)
                else:
                    lazy_field = getattr(self, f.name)

                lazy_field.field_name = f.name
                lazy_field.type = f.metadata["type"]

    def set_folder(self, folder):
        self._folder = folder
        for f in dataclasses.fields(self):
            if f.type == LazyField:
                getattr(self, f.name).folder = folder

    def serialize(self, folder=None):
        if folder:
            self.set_folder(folder)

        for f in dataclasses.fields(self):
            if f.type == LazyField:
                getattr(self, f.name).save()

        return self.to_json()

    @classmethod
    def deserialize(cls, json_str, folder):
        result = cls.from_json(json_str)
        result.set_folder(folder)
        return result

    @property
    def folder(self):
        return Path(self._folder)


def generate_flyte_result_class(dc_class: BaseResultDC):
    """Generate flyte result class from a given result dataclass.

    Arguments:
        dc_class: can be any result dataclass that inherits from BaseResultDC

    Returns:
        FlyteResult: class that wraps a result dataclass for flyte.
    """

    @dataclass
    class FlyteResult(DataClassJsonMixin):
        result: dc_class
        dir: FlyteDirectory

        def load_result(self) -> dc_class:
            self.result.set_folder(Path(self.dir.download()))
            return self.result

        def save_to_disk(self, path: Path, force: bool = False):
            """
            Writes the result to a folder
            """
            # Create the folder if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)

            # Write out the result dataclass
            (path / "result.yaml").write_text(yaml.safe_dump(self.result.to_dict()))

            # Specify the path for the result data
            data_dir = path / "data"

            # Download the flytedirectory
            tmp_dir = self.dir.download()

            # Copy it to the destination
            shutil.copytree(tmp_dir, data_dir, dirs_exist_ok=force)

        def __repr__(self):
            return f"FlyteResult ({type(self.result)} @ {self.dir.path})"

        @classmethod
        def load_from_disk(self, path: Path) -> "FlyteResult":
            """
            Loads a result from disk
            """
            try:
                result_path = path / "result.yaml"
                result_dict = yaml.safe_load(result_path.read_text())
            except FileNotFoundError:
                # Result files used to have the json extension and internal format yaml
                result_path = path / "result.json"
                result_dict = yaml.safe_load(result_path.read_text())

            result = dc_class.from_dict(result_dict)

            return FlyteResult(result, FlyteDirectory(str(path / "data")))

        @classmethod
        def from_inst(cls, inst):
            if inst._folder is None:
                working_dir = flytekit.current_context().working_directory
                pp = Path(working_dir) / "res"
                pp.mkdir(exist_ok=True)

                inst.serialize(pp)
            else:
                inst.serialize()

            return cls(result=inst, dir=FlyteDirectory(str(inst.folder)))

    FlyteResult.__name__ = "Flyte" + dataclass.__name__

    return FlyteResult


def lazy_field(field_type):
    return field(default_factory=LazyField, metadata=dict(type=field_type))
