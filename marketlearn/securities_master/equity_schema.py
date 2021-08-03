# pyre-strict
"""Module creates the Securitities Master DataBase for Equities

Author: Rajan Subramanian
Date: 11/16/2020
"""


from dataclasses import dataclass, field
from typing import List

from marketlearn.dbhelper import DbReader


@dataclass
class EquitySchema:
    """Creates Schema to hold end of day equity prices

    Parameters
    ----------
    None

    Attributes
    ----------
    exchange : str
        exchange table lists exchange we wish to obtain equity
        pricing information from.
        Currently only supports NYSE and NASDAQ
    data_vendor : str
        this table lists information about historical pricing data
        vendors.  Yahoo EOD data will be used
    security : str
        security table stores list of ticker symbols and company information
    eq_daily_price : str
        this table stores daily pricing information for each security
    """

    exchange: str = field(init=False)
    data_vendor: str = field(init=False)
    security: str = field(init=False)
    eq_daily_price: str = field(init=False)
    _db: DbReader = field(init=False)

    def __post_init__(self) -> None:
        self.exchange = "exchange"
        self.data_vendor = "data_vendor"
        self.security = "security"
        self.eq_daily_price = "eq_daily_price"
        self._db = DbReader()

    def create_exchange_query(self) -> str:
        """creates the exchange table"""
        query = """Create table exchange(
                id serial,
                abbrev varchar(32) not null,
                name varchar(255) not null,
                city varchar(255) null,
                currency varchar(64) null,
                timezone_offset time null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                constaint pk_exchange primary key (id)
                );
            """
        return query

    def create_data_vendor_query(self) -> str:
        """creates the data_vendor table"""
        query = """create table data_vendor(
                id serial,
                name varchar(64) not null,
                website_url varchar(255) null,
                support_email varchar(255) null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                constraint pk_vendor primary key (id)
            );
            """
        return query

    def create_security_query(self) -> str:
        """creates the symbol table"""
        query = """create table security(
                id serial,
                exchange_id int not null,
                ticker varchar(32) not null,
                instrument varchar(64) not null,
                name varchar(255) null,
                sector varchar(255) null,
                currency varchar(32) null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                constraint fk_exchange_id
                  foreign key (exchange_id) references exchange (id)
                constraint pk_symbol primary key (id)
            );
            """
        return query

    def create_eq_daily_price_query(self) -> str:
        """creates daily equity price information"""
        query = """create table eq_daily_price(
                id serial,
                data_vendor_id int not null,
                security_id int not null,
                price_date timestamp not null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                open_price numeric(19,4) null,
                high_price numeric(19,4) null,
                low_price numeric(19,4) null,
                close_price numeric(19,4) null,
                adj_close_price numeric(19,4) null,
                volume bigint null,
                constraint fk_data_vendor_id
                  foreign key (data_vendor_id) references data_vendor (id),
                constraint fk_security_id
                  foreign key (security_id) references security (id),
                constraint pk_eq_daily_price primary key (id)
                );
            """
        return query

    def clear_tables(self, table_names: List[str]) -> None:
        """clears tables from database given by table_names

        Parameters
        ----------
        table_names : List[str]
            clear the tables but doesn't delete the schema
        """
        queries = (f"delete from {table_name}" for table_name in table_names)
        self._db.execute(queries)

    def create_tables(self) -> None:
        """creates tables"""
        # create a list of table queries
        queries = (
            self.create_exchange_query(),
            self.create_security_query(),
            self.create_data_vendor_query(),
            self.create_eq_daily_price_query(),
        )

        # create the tables
        self._db.execute(queries)
