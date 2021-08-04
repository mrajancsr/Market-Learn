# pyre-strict
"""
PostGresSQL python Interface for interacting with Dbeaver DB
Author: Rajan Subramanian
Created May 10 2020
Todo - add a copy from
"""

import os
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import pandas as pd
import psycopg2
from marketlearn.toolz import timethis
from psycopg2.extras import DictCursor, execute_values

connection = TypeVar("connection")


class PathNotFoundError(Exception):
    pass


@dataclass
class DbReader(Generic[connection]):
    """Establishes a sql connection with the PostGres Database

    Parameters
    ----------
    section: str, default='postgresql-dev'
        one of postgresql-dev or postgresql-practice


    Attributes
    ----------
    conn: Optional[psycopg2.connection]
        connection objevct for psycopg2
    """

    section: str = "dev"
    conn: Optional[connection] = field(init=False)
    column_names: Set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.conn = None
        self.column_names = {""}

    def _read_db_config(self) -> Dict[str, str]:
        """reads database configuration from config.ini file

        Parameters
        ----------
        section : str, optional, default='postgresql-dev'
            one of postgresql-dev or postgresql-practice

        Returns
        -------
        Dict
            contains database configuration

        Raises
        ------
        Exception
            raises error if wrong environment is chosen
        """
        # create the parser
        file_name = "config.ini"
        config_path = os.getenv("SQLCONFIGPATH")
        if isinstance(config_path, str):
            file_path = os.path.join(config_path, file_name)
            file_path = file_path.replace(":", "")
        else:
            raise PathNotFoundError(
                "SQLCONFIGPATH is incorrect - please check"
            )

        parser = ConfigParser()
        parser.read(file_path)

        # get the section, default to postgressql
        section = "postgresql-" + self.section
        config = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                config[param[0]] = param[1]
        else:
            raise Exception(
                f"{section} not found in the {file_name} \
                \nfilepath: {file_path}"
            )
        return config

    def connect(self) -> Optional[connection]:
        """Connects to PostGresSql Database

        Parameters
        ----------
        section : str, optional, default='dev'
            one of dev or practice

        Returns
        -------
        connection object
            connection to the PostGresSQL DB
        """
        connection_closed = getattr(self.conn, "closed")
        if self.conn is None or connection_closed:
            try:
                params = self._read_db_config()
                self.conn = psycopg2.connect(**params)
                return self.conn
            except psycopg2.DatabaseError as error:
                print(error)

    def _create_records(
        self, dictrow: Tuple[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], ...]:
        """converts data obtained from db into tuple of dictionaries"""
        return tuple({k: v for k, v in record.items()} for record in dictrow)

    def fetch(self, query: str) -> Optional[Tuple[Dict[str, Any]]]:
        """Returns the data associated with table
        Args:
        query:  database query parameter
        records:   specify if the rows should be returned as records

        Returns:
        list of DictRows where each item is a dictionary
        """

        try:
            self.connect()
            conn = getattr(self, "conn")
            if conn:
                column_names = set()
                with conn.cursor(cursor_factory=DictCursor) as curr:
                    curr.execute(query)
                    for col in curr.description:
                        column_names.add(col.name)
                    rows = curr.fetchall()
                getattr(conn, "close")()
                self.column_names = column_names
        except psycopg2.DatabaseError as error:
            print(error)
            getattr(conn, "close")()
        else:
            return rows

    @timethis
    def fetchdf(self, query: str) -> pd.DataFrame:
        """Returns a pandas dataframe of the db query"""
        return pd.DataFrame(self.fetch(query), columns=self.column_names)

    def iterator_from_df(
        self, datadf: pd.DataFrame
    ) -> Iterator[Dict[str, Any]]:
        """Convenience function to transform pandas dataframe to
        Iterator for db push
        """
        return iter(datadf.to_dict(orient="rows"))

    def push(
        self,
        data: Iterator[Dict[str, Any]],
        table_name: str,
        columns: List[str],
    ) -> None:
        """Pushes data given as an Iterator to PostGresDB

        :param data: data to be pushed
        :type data: Iterator[Dict[str, Any]]
        :param table_name: the table to insert data into
        :type table_name: str
        :param columns: the columns where data is inserted to
        :type columns: List[str]
        :param section: currently only supports 'dev',
         defaults to 'dev'
        :type section: str, optional
        """
        try:
            self.connect()
            with self.conn.cursor() as curr:
                # get the column names
                col_names = ",".join(columns)

                # create an iterator of tuple rows for db insert
                args = (tuple(item.values()) for item in data)
                query = f"""INSERT INTO {table_name} ({col_names}) values %s"""

                # insert data into database and close connection
                execute_values(curr, query, args)
            self.conn.commit()
            self.conn.close()
        except psycopg2.DatabaseError as e:
            print(e)

    @timethis
    def pushdf(
        self,
        datadf: pd.DataFrame,
        table_name: str,
    ) -> None:
        """Pushes pandas dataframe to database

        :param datadf: data to be pushed
        :type datadf: pd.DataFrame
        :param table_name: the name of table to be pushed to
        :type table_name: str
        :param section: currently only supports 'dev',
        defaults to 'dev'
        :type section: str, optional
        """
        # get the column names
        cols = list(datadf)

        # create an iterator from pandas df prior to push
        data = self.iterator_from_df(datadf)

        # push the data to table given by table_name
        self.push(data, table_name=table_name, columns=cols)

    def copy_from(
        self,
        data: Iterator[Dict[str, Any]],
        table_name: str,
        columns: List[str],
        section: str = "dev",
    ) -> None:
        """copies data from csv file and writes to Db"""
        raise NotImplementedError("Will be Implemented Later")

    def drop(self, table_name: str, conn) -> None:
        """removes table given by table_name from dev db"""
        query = f"drop table {table_name};"
        self.execute(query, conn)

    def delete(self, table_name):
        """deletes all rows given by table_name from dev deb
        table schema is retained
        """
        query = f"delete from {table_name};"
        self.execute(query)

    def execute(self, query: Union[str, List[str]]):
        """executes a sql query statement

        Parameters
        ----------
        query : Union[str, List[str]]
            can be either a statement or list of queries
        section : str, optional, default='dev'
            supposed one of dev or practice
        """
        try:
            self.connect()
            with self.conn.cursor() as curr:
                if isinstance(query, str):
                    curr.execute(query)
                elif isinstance(query, list):
                    for q in query:
                        curr.execute(q)
            self.conn.commit()
            self.conn.close()
        except Exception as e:
            print(e)


db = DbReader()
