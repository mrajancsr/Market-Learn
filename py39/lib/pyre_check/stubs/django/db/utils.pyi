# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Iterable

from django.db.backends.base.base import BaseDatabaseWrapper

class ConnectionHandler:
    def __init__(self, databases=None): ...
    def __getitem__(self, alias) -> BaseDatabaseWrapper: ...
    def close_all(self): ...
    def all(self) -> Iterable[BaseDatabaseWrapper]: ...

class Error(Exception): ...
class InterfaceError(Error): ...
class DatabaseError(Error): ...
class DataError(DatabaseError): ...
class OperationalError(DatabaseError): ...
class IntegrityError(DatabaseError): ...
class InternalError(DatabaseError): ...
class ProgrammingError(DatabaseError): ...
class OperationalError(DatabaseError): ...
class NotSupportedError(DatabaseError): ...