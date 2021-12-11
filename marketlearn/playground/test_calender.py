import unittest
from requests.exceptions import Timeout
from unittest.mock import Mock
from parameterized import parameterized
from nose.tools import assert_equal
import math

requests = Mock()


def get_holidays():
    r = requests.get("http://localhost/api/holidays")
    if r.status_code == 200:
        return r.json()
    return None


class TestCalendar(unittest.TestCase):
    def test_get_holidays(self):
        requests.get.side_effect = Timeout
        with self.assertRaises(Timeout):
            get_holidays()

    def log_request(self, url):
        print(f"Making request to {url}.")
        print("Request Received")

        # Create a new Mock to imitate a response
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "12/25": "Christmas",
            "7/4": "Indepdendence dAy",
        }
        return response_mock

    def test_get_holidays_logging(self):
        requests.get.side_effect = self.log_request
        self.assertEqual(get_holidays()["12/25"], "Christmas")

    # above methods isn't that good.  t his way is much better
    def test_get_holidays_retry(self):
        "create a new mock to imitate ar esponse"
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "12/25": "Christmas",
            "7/4": "Indepdendence dAy",
        }
        # set the side effect of get
        requests.get.side_effect = [Timeout, response_mock]
        # test that first request resulted in timeout
        with self.assertRaises(Timeout):
            get_holidays()
        # now retry expecting a successful outcome
        self.assertEqual(get_holidays()["12/25"], "Christmas")
        # test if .get() was called twice
        assert requests.get.call_count == 4

    @parameterized.expand(
        [
            (2, 2, 4),
            (2, 3, 8),
            (1, 9, 1),
            (0, 9, 0),
        ]
    )
    def test_pow(self, base, exponent, expected):
        print("running test")
        print(base, exponent, expected)
        self.assertEqual(math.pow(base, exponent), expected)


unittest.main()