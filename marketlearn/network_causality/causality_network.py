"""
Implementation of Econometric measures of
connectness and systemic risk in finance and
insurance sectors by M.Billio, M.Getmansky,
Andrew Lo, L.Pelizzon
"""

from typing import Dict
from itertools import combinations, product
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
from scipy.sparse.linalg import eigs
from sector_data import SectorPrice
from mlfinlab.network_causality.vector_ar.bivar import BiVariateVar
from mlfinlab.network_causality.graph import GraphAdMap

# pylint: disable=invalid-name, undefined-loop-variable

class CNet:
    """
    Class Implements the granger causal flows
    in a complicated network
    """
    class PreProcess:
        """
        Nested Class to pre-process data
        before program start and create
        "sectors" attribute
        """
        def __init__(self, start: str = '1999-12-31'):
            """
            Constructor used to instantiate class
            :param start: (str) start date in format 'YYYY-MM-DD'
            """
            self.start = start
            self.preprocess_sectors()
            self.transformed_data = None

        @staticmethod
        def _get_sectors() -> pd.DataFrame:
            """
            Downloads the sector prices from SectorPrice
            :return: (pd.DataFrame) *?
            """
            return SectorPrice().read()

        def preprocess_sectors(self):
            """
            Preprocesses data by removing any NAs
            and create sectors attribute
            :return: (None)
            """
            # pylint: disable=consider-iterating-dictionary

            sec = self._get_sectors()
            for k in sec.keys():
                sec[k] = sec[k][sec[k].index >= self.start].dropna(axis=1)

            # Create the sectors attribute
            self.sectors = sec.copy()

            # Garbage collection
            sec = None

        def get(self) -> pd.DataFrame:
            """
            Returns the sectors after preprocessing
            """
            return self.sectors

    # --- main class definition

    def __init__(self, start: str):
        self.sectors = self.PreProcess(start=start).get()
        self.pca = PCA()
        self.sc = StandardScaler()
        self.lr = None
        self.ret1 = None
        self.ret2 = None
        self.errors = None
        self.transformed_data = None

    def risk_fraction(self, data: pd.DataFrame, n: int = 3):
        """
        Computes the cumulative risk fraction of system
        see ref: formula (6) of main paper
        :param data: (pd.DataFrame) end of month prices
            shape = (n_samples, p_shares)
        :param n: (int) Number of principal components (3 by default)
            assumes user has chosen the best n
        :return: (float)
        """
        # Store col names
        col_names = list(data)

        # Compute log returns
        data = np.log(1 + data.pct_change())
        data = self.sc.fit_transform(data.dropna())
        data = self.pca.fit_transform(data)
        self.transformed_data = pd.DataFrame(data, columns=col_names)

        # Total risk of system
        system_risk = np.sum(self.pca.explained_variance_)

        # Risk associated with first n principal components
        pca_risk = self.pca.explained_variance_[:n].sum() / system_risk

        return pca_risk

    def is_connected(self,
                     data: pd.DataFrame,
                     n: int = 3,
                     thresh: float = 0.3) -> bool:
        """
        Determines the interconnectedness in a system
        see ref: formula (6) of main paper
        :param data: (pd.DataFrame) end of month prices
        :param n: (int) Number of principal components (3 by default)
        :param thresh: (int) Prespecified threshold (0.3 by default)
        :return: (bool) True if first n principal components
            explains more than some fraction thresh of total volatility
        """
        return self.risk_fraction(data, n=n) >= thresh

    def pcas(self,
             data: pd.DataFrame,
             institution_i: str,
             n: int = 3,
             thresh: float = 0.3) -> pd.Series():
        """
        Measures connectedness for each company
        or exposure of company to total risk of system
        see ref: formula (8)
        :param data: (pd.DataFrame) end of month prices
        :param institution_i: (str) name of the institution
        :param n: (int) Number of principal components (3 by default)
        :param thresh: (int) Prespecified threshold (0.3 by default)
        :return: (pd.Series) if institution_i is None, return
            the connectedness of each company to system as a series
            otherwise returns the exposure of institution_i
        """
        if not self.is_connected(data, n, thresh):
            raise ValueError("system not connected - increase n or thresh")

        # Get the variances of each institution
        var = self.transformed_data.var()

        # Get system variance
        system_var = self.transformed_data.cov().sum().sum()

        # Get the loadings
        loadings = self.pca.components_[:n] ** 2
        weights = self.pca.explained_variance_[:n]
        result = (weights @ loadings).sum() * var / system_var
        return result if institution_i is None else result[institution_i]

    def linear_granger_causality(self,
                                 data1: pd.Series(),
                                 data2: pd.Series(),
                                 alpha: float = 0.05
                                 ) -> dict:
        """
        Tests if data1 granger causes data2
        :param data1: (pd.Series)
        :param data2: (pd.Series) *?
        :param alpha: (float) *? (0.05 by default)
        :return: (dict) containing True, False result of
            causality. Key1='x_granger_causes_y', key2='y_granger_causes_x'
        """
        # Log prices pt = log(Pt)
        logp1 = np.log(data1).values
        logp2 = np.log(data2).values

        # Log returns rt = pt - pt-1
        ret = np.diff(logp1)
        ret2 = np.diff(logp2)

        # Remove mean from sample prior to garch fit
        returns = [None, None]
        # g = Garch(mean=False)
        idx = 0
        for r in [ret, ret2]:
            _, pval = ttest_1samp(r, 0)
            # Sample mean is not zero
            if pval < 0.05:
                r -= r.mean()
            #g.fit(r)
            am = arch_model(100*r, mean='Zero')
            res = am.fit(disp='off')
            returns[idx] = r / res.conditional_volatility
            idx += 1

        # Scaled returns based on garch volatility
        ret, ret2 = returns

        # Check for cointegration
        bivar = BiVariateVar(fit_intercept=True)
        self.lr = bivar.lr
        coint = bivar.coint_test(logp1, logp2, alpha=alpha)
        self.ret1 = ret
        self.ret2 = ret2

        # Auto select based on lowest bic and fit
        bivar.select_order(ret, ret2, coint=coint)

        # Check for granger causality
        result = bivar.granger_causality_test()

        return result

    def _create_casual_network(self,
                               data: pd.DataFrame()
                               ) -> GraphAdMap:
        """
        Creates connections between N financial Institutions
        :param data: (pd.DataFrame) end of month prices
        :return: (GraphAdMap) graph of adjacency map
            containing causality network between institutions
            in data
        """
        # Create a directed graph
        g = GraphAdMap(directed=True)
        share_names = iter(list(data))
        vertices = iter([g.insert_vertex(v) for v in share_names])

        # Create granger causality network
        key1 = 'x_granger_causes_y'
        key2 = 'y_granger_causes_x'
        for c in combinations(vertices, 2):
            # Extract the vertices
            u, v = c

            # Get the respestive prices
            price1 = data[u.get_value()]
            price2 = data[v.get_value()]
            try:
                # Check for linear granger causality
                granger_result = self.linear_granger_causality(price1, price2)
                if granger_result[key1]:
                    g.insert_edge(u, v, 1)
                if granger_result[key2]:
                    g.insert_edge(v, u, 1)
            except ValueError as e:
                self.errors = []
                print("Error occured for {}".format(e))
                self.errors.append((u.get_value(), v.get_value()))
        return g

    def _create_sector_casual_network(self,
                                      sector1: pd.DataFrame(),
                                      sector2: pd.DataFrame()
                                      ) -> GraphAdMap:
        """
        Creates connections between instituions in sector 1
        :param sector1: (pd.DataFrame) Monthly prices of securities
        :param sector2: (pd.DataFrame) Monthly prices of securities
        :return: (GraphAdMap) Graph whose vertex are share names
            among the two sectors and edges represent granger causality
        """
        # Create a directed graph
        g = GraphAdMap(directed=True)

        # Create sector vectors
        sector1_vectors = iter([g.insert_vertex(v) for v in list(sector1)])
        sector2_vectors = iter([g.insert_vertex(v) for v in list(sector2)])

        # Create granger causality network
        key1 = 'x_granger_causes_y'
        key2 = 'y_granger_causes_x'
        for v in product(sector1_vectors, sector2_vectors):
            # Extract the vertices
            sec1, sec2 = v
            # We don't want same companies in different sectors
            if sec1.get_value() != sec2.get_value():
                # Get the respective prices
                price1 = sector1[sec1.get_value()]
                price2 = sector2[sec2.get_value()]

                # Check for granger causality
                granger_result = self.linear_granger_causality(price1, price2)
                if granger_result[key1]:
                    g.insert_edge(sec1, sec2, 1)
                if granger_result[key2]:
                    g.insert_edge(sec2, sec1, 1)
        return g

    @staticmethod
    def granger_causality_degree(data: pd.DataFrame(),
                                 graph: GraphAdMap
                                 ) -> float:
        """
        Computes ratio of statistically significant Granger
        causality relationships among N(N-1) pairs of
        N financial institutions
        :param data: (pd.DataFrame) end of month prices
        :param graph: (GraphAdMap) causality network
        :return: (float) fraction causality
        """

        n = data.shape[1]
        count = 0

        for e in graph.get_edges():
            if e.get_value() == 1:
                count += 1

        return count / (n * (n - 1))

    def number_of_connections(self,
                              data: pd.DataFrame(),
                              institution_name: str = 'AAPL',
                              thresh: float = 0.1,
                              conn_type: str = 'out',
                              graph: GraphAdMap = None
                              ) -> float:
        """
        *?
        :param data: (pd.DataFrame) end of month prices
        :param institution_name: (str) company name in data
        :param thresh: (float) granger threshold parameter
        :param conn_type: (int) in-out and total connections
        :return: (float) fraction of companies granger caused
            to/by institution name given by conn_type
        """

        result = None

        # Create graph if it doesn't exist
        if graph is None:
            graph = self._create_casual_network(data)

        # Get the dgc score
        dgc = self.granger_causality_degree(data, graph=graph)
        n = data.shape[1]
        if dgc >= thresh:
            # Find the institution in the network
            for v in graph.get_vertices():
                if v.get_value() == institution_name:
                    break

            if v.get_value() != institution_name:
                raise ValueError("institution name not found")

            if conn_type == 'out':
                result = graph.degree(v) / (n - 1)
            elif conn_type == 'in':
                result = graph.degree(v, outgoing=False) / (n - 1)
            elif conn_type == 'total':
                total = graph.degree(v) + graph.degree(v, outgoing=False)
                result = total / (2*(n - 1))

        return result

    def sector_connections(self,
                           data: Dict[str, pd.DataFrame],
                           sector_name: str = 'bdealer',
                           institution_name: str = 'AAPL',
                           conn_type: str = 'out',
                           ) -> float:
        """
        Computes sector-conditional connections
        :param data: (Dict[str, pd.DataFrame]) Dictionary of sector prices
            key is one of ['bdealer','bank','hedge_fund','insurance']
            value is dataframe of monthly prices
        :param sector_name: (str) Represents key to data ('bdealer' by defrault)
        :param institution_name: (str) Name of institution in sector_name
            that granger causes institutions in other sectors ('AAPL' by default)
        :param conn_type: (str) in and out connections
            supports 'in', 'out' and 'total'
        :return: (float) fraction of sector connections
        """
        # Get total sectors and sum of all institutions
        m = len(data.keys())
        n = sum(map(lambda x: x.shape[1], data.values()))

        # Denominator in pg. 12
        denom = (n / m) * (m - 1)

        # The sector we are comparing with
        key = data[sector_name]
        count = 0
        for sector in data.keys() - {sector_name}:
            g = self._create_sector_casual_network(key, data[sector])
            for v in g.get_vertices():
                if v.get_value() == institution_name:
                    break
            if v.get_value() != institution_name:
                raise ValueError("Institution name not found")

            if conn_type == 'out':
                count += g.degree(v)
            elif conn_type == 'in':
                count += g.degree(v, outgoing=False)
            elif conn_type == 'total':
                count += g.degree(v) + g.degree(v, outgoing=False)

        # Return the count
        if conn_type  not in ('out', 'in', 'total'):
            raise AttributeError("incorrect connection type")

        if conn_type in ('out', 'in'):
            result = count / denom
        elif conn_type == 'total':
            result = count / (2 * denom)

        return result

    @staticmethod
    def _search(graph: 'GraphAdMap',
                start: 'GraphAdMap._Vertex',
                discovered: Dict
                ):
        """
        Performs a breadth first search on undiscovered portion
        of GraphAdMap graph starting at vertex start
        Updates the discovered dictionary
        :param graph: (GraphAdMap) Adjacency Map containing causal network
        :param start: (GraphAdMap._Vertex) vertex to start teh search from
        :param discovered: (Dict) if a vertex is already visited, its added
            to this dictionary
        """
        # Beginning only includes start
        start_search = [start]
        while len(start_search) > 0:
            connected_institutions = []
            for institution in start_search:
                for e in graph.iter_incident_edges(institution):
                    connected = e.opposite(institution)
                    if connected not in discovered:
                        discovered[connected] = e
                        connected_institutions.append(connected)
            start_search = connected_institutions

    @staticmethod
    def _construct_gc_path(start: 'GraphAdMap._Vertex',
                           end: 'GraphAdMap._Vertex',
                           discovered: Dict
                           ) -> list:
        """
        Constructs Granger Causality path between two institutions
        given by start and end parameter
        :param graph: (GraphAdMap) adjacency map that contains causal network
        :param start: (GraphAdMap._Vertex) vertex to start the search from
        :param end: (GraphAdMap._Vertex) vertex to end the search
        :param discovered: (Dict) holds the vertices that have been visited
        :return: (list) path between two institutions
        """
        path = []
        if end in discovered:
            path.append(end)
            walk = end
            while walk is not start:
                connected = discovered[walk]
                parent = connected.opposite(walk)
                path.append(parent)
                walk = parent
            path.reverse()
        return path

    def closeness(self,
                  data: pd.DataFrame(),
                  institution_name: str,
                  thresh: float = 0.1,
                  graph: GraphAdMap = None
                  ) -> float:
        """
        Measures shortest path between financial institution
        and all other institutions reachable from it
        Raises:
            ValueError: If instiution name is not found
        :param data: (pd.DataFrame) end of month prices
        :param institution_name (str): name of company in data.
        :param thresh (float): threshold of causality score.
            Defaults to 0.1.
        :param graph (GraphAdMap, optional): granger causal network
            that holds causality connections between institutions
        :return: (float) *?
        """

        # Create causal network if not supplied
        if graph is None:
            graph = self._create_casual_network(data)
        dgc = self.granger_causality_degree(data, graph=graph)

        # Discovered institutions connected to institution_name
        discovered = {}
        count = 0
        n = data.shape[1]
        if dgc >= thresh:
            # Find the institution in the network
            for v in graph.get_vertices():
                if v.get_value() == institution_name:
                    break
            if v.get_value() != institution_name:
                raise ValueError("institution name not found")
            # Start the search from v to all other institutions
            self._search(graph, v, discovered)
            # Count the number of paths
            start = v
            for end in graph.get_vertices() - {start}:
                path = self._construct_gc_path(start, end, discovered)
                if path:
                    count += 1
        return count / (n-1)

    def eigen_vector_centrality(self,
                                data: pd.DataFrame(),
                                institution_name: str = 'AAPL',
                                thresh: float = 0.1):
        """
        Computes the evc of institution given by inst.name
        :param data: (pd.DataFrame) *?
        :param institution_name: (str) *? ('AAPL' by default)
        :param thresh: (float) *? (0.1 by default)
        :return: (*?) *?
        """
        result = None

        # Get the dgc
        dgc = self.granger_causality_degree(data, None)
        if dgc >= thresh:
            # Create the network
            g = self._create_casual_network(data)

            # Get the adjacency matrix
            adj_matrix = g.get_adjacency_matrix()

            # Get the eigen vector corresponding to evalue of 1
            _, evector = eigs(adj_matrix, k=1, sigma=1.0)
            df = pd.DataFrame(adj_matrix, columns=data.columns)
            df.index = data.columns
            result = (df.loc[institution_name, ] * evector).sum()

        return result