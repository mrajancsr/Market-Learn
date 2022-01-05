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
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor, execute_values

# Custom Type to represent psycopg2 connection
connection = TypeVar("connection")


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
    conn: Optional[connection] = None
    column_names: List[str] = field(init=False, default_factory=list)

    def _read_db_config(self) -> Dict[str, str]:
        """reads database configuration from config.ini file

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

        if os.path.exists(config_path):
            file_path = os.path.join(config_path, file_name)
        else:
            raise FileNotFoundError(f"config_path: {config_path}")

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
            raise Exception(f"Incorrect Section: {section}")
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

        try:
            params = self._read_db_config()
            conn = psycopg2.connect(**params)
            return conn
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
            conn = self.connect()
            if conn:
                column_names = set()
                with conn.cursor(cursor_factory=DictCursor) as curr:
                    curr.execute(query)
                    for col in curr.description:
                        column_names.add(col.name)
                    rows = curr.fetchall()
                conn.close()
                self.column_names = column_names
                return rows
        except psycopg2.DatabaseError as error:
            print(error)

    def fetchdf(self, query: str) -> pd.DataFrame:
        """Returns a pandas dataframe of the db query"""
        return pd.DataFrame(self.fetch(query), columns=self.column_names)

    def iterator_from_df(self, datadf: pd.DataFrame) -> Iterator[Dict[str, Any]]:
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
        """Pushes data to postgresql database

        Parameters
        ----------
        data : Iterator[Dict[str, Any]]
            to push to postgresql database
        table_name : str
            the table to push to
        columns : List[str]
            column names of the data

        Raises
        ------
        psycopg2.DatabaseError
            if the table doesn't exist or incorrect data format
        """
        try:
            self.connect()
            conn = getattr(self, "conn")
            if conn:
                with conn.cursor() as curr:
                    # get the column names
                    col_names = ",".join(columns)

                    # create an iterator of tuple rows for db insert
                    args = (tuple(item.values()) for item in data)
                    query = f"""INSERT INTO {table_name} ({col_names}) values %s"""

                    # insert data into database and close connection
                    execute_values(curr, query, args)
                conn.commit()
                conn.close()
        except psycopg2.DatabaseError as e:
            print(e)

    def pushdf(
        self,
        datadf: pd.DataFrame,
        table_name: str,
    ) -> None:
        """Pushes pandas dataframe to postgresql database

        Parameters
        ----------
        datadf : pd.DataFrame
            data to push to database
        table_name : str
            the table to push to
        """
        column_names = list(datadf)
        data = self.iterator_from_df(datadf)
        self.push(data, table_name=table_name, columns=column_names)

    def copy_from(
        self,
        data: Iterator[Dict[str, Any]],
        table_name: str,
        columns: List[str],
        section: str = "dev",
    ) -> None:
        """copies data from csv file and writes to Db"""
        raise NotImplementedError("Will be Implemented Later")

    def drop(self, table_name: str) -> None:
        """removes table given by table_name from dev db

        Parameters
        ----------
        table_name : str
            the table in database
        """
        self.execute(f"drop table {table_name};")

    def delete(self, table_name: str) -> None:
        """[summary]

        Parameters
        ----------
        table_name : [type]
            [description]
        """
        self.execute(f"delete from {table_name};")

    def execute(self, query: Union[str, Tuple[str]]) -> None:
        """executes a sql query statement

        Parameters
        ----------
        query : Union[str, Tuple[str]]
            can be either a statement or tuple of queries
        """
        try:
            self.connect()
            conn = getattr(self, "conn")
            if conn:
                with conn.cursor() as curr:
                    if isinstance(query, str):
                        curr.execute(query)
                    elif isinstance(query, tuple):
                        for q in query:
                            curr.execute(q)
                conn.commit()
                conn.close()
        except Exception as e:
            print(e)


def main():
    db = DbReader()
    query = "select * from customer"
    df = db.fetchdf(query)
    print(df.head())
    data = db.fetch(query)
    print(data)


if __name__ == "__main__":
    db = DbReader()
    query = "select * from customer"
    df = db.fetchdf(query)
    df.head()

main()